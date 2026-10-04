# -*- coding: utf-8 -*-
"""
metodo3_ablacao.py

Traceback token-level para features SAE via ablacao causal por posicao (Metodo 3).

Dada uma feature SAE identificada pelo FADEx, este modulo responde: quais tokens
do texto CAUSALMENTE sustentam a ativacao dessa feature? Diferente do Gradient x
Input (Metodo 2, que mede saliencia via aproximacao linear de primeira ordem), a
ablacao mede causalidade genuina: zera a contribuicao de cada token e observa o
quanto a ativacao da feature (no last-token) cai.

    attr(j) = h(x)_i - h(x)_i^{(-j)}

onde h(x)_i^{(-j)} e a ativacao da feature com o token j ablacionado. Quanto maior
a queda, mais causalmente importante o token j.

Decisoes de design (ver discussao no pipeline):
    - Remocao via ATTENTION MASKING: zera-se a atencao PARA a posicao j, mantendo
      o comprimento da sequencia. Nao ha delecao fisica nem re-indexacao posicional
      (que quebraria o RoPE e desalinharia o last-token pooling).
    - Execucao BATCHED: para ablacionar T posicoes, replica-se o texto T vezes num
      batch; cada linha zera uma posicao distinta da attention_mask. Um forward
      resolve todas as ablacoes. Sub-batching evita estouro de VRAM em textos longos.
    - Posicao de LEITURA (last-token) nao e ablacionada: e de onde se le a ativacao.
    - BOS por padrao nao e ablacionado (skip_bos=True): ablacionar o attention sink
      colapsa a redistribuicao de atencao e mede o colapso do mecanismo, nao
      importancia semantica. Configuravel.

Distribution shift: mascarar a atencao para um token cria um contexto que o modelo
nao viu exatamente no treino. E uma limitacao inerente da ablacao; por isso o Metodo
3 e recomendado como VALIDACAO do Metodo 2, nao como pipeline unico.

Convencoes herdadas do pipeline: left padding + last-token pooling, encoder SAE em
fp32 (sensibilidade do threshold JumpReLU), Gemma 2 2B em bf16.

Reutiliza carregadores e utilitarios de metodo2_traceback (DifferentiableSAEEncoder
nao e necessario aqui — ablacao usa so forward, sem gradiente — mas o encoder em
modo no_grad e reaproveitado para computar a ativacao da feature).
"""

import os
import json
import contextlib

import numpy as np
import torch


# =============================================================================
# 1. Encoder SAE (apenas forward, sem gradiente)
# =============================================================================

class SAEEncoderForward:
    """
    Encoder JumpReLU somente-forward para ablacao. Nao precisa de gradiente
    (a ablacao mede diferencas de ativacao entre forwards, nao derivadas).

    Opera em batch: recebe (B, d_model) e retorna (B,) com a ativacao da feature
    para cada item — util porque a ablacao batched avalia T contextos de uma vez.
    """

    def __init__(self, W_enc: torch.Tensor, b_enc: torch.Tensor, theta: torch.Tensor):
        self.W_enc = W_enc          # (d_model, d_sae)
        self.b_enc = b_enc          # (d_sae,)
        self.theta = theta          # (d_sae,)

    @torch.no_grad()
    def feature_activation(self, x: torch.Tensor, feature_idx: int) -> torch.Tensor:
        """
        Ativacao JumpReLU da feature para um batch de residual streams.
        x: (B, d_model) fp32. Retorna (B,) fp32.
        """
        # So precisamos da coluna feature_idx do encoder: projecao (d_model,) -> escalar
        w_i   = self.W_enc[:, feature_idx]                 # (d_model,)
        pre_i = x @ w_i + self.b_enc[feature_idx]          # (B,)
        gate  = (pre_i > self.theta[feature_idx]).float()  # (B,)
        return pre_i * gate


def load_sae_encoder_forward(sae_repo: str, sae_path: str,
                             d_model: int, d_sae: int, device: str):
    """Baixa os pesos da SAE e monta o encoder somente-forward (fp32)."""
    from huggingface_hub import hf_hub_download

    sae_file = hf_hub_download(repo_id=sae_repo, filename=sae_path)
    params   = np.load(sae_file)

    W_enc = torch.from_numpy(params["W_enc"]).float().to(device)
    b_enc = torch.from_numpy(params["b_enc"]).float().to(device)
    theta = torch.from_numpy(params["threshold"]).float().to(device)

    assert W_enc.shape == (d_model, d_sae), \
        f"W_enc shape {tuple(W_enc.shape)} != esperado ({d_model}, {d_sae})"

    return SAEEncoderForward(W_enc, b_enc, theta)


# =============================================================================
# 2. Motor de ablacao causal (batched, attention masking)
# =============================================================================

class AblationEngine:
    """
    Executa ablacao causal por posicao de uma feature SAE em um texto.

    Estrategia batched: para um texto com T tokens, monta um batch onde a linha 0
    e o texto intacto (baseline) e as linhas 1..T tem cada uma zerada a atencao
    para uma posicao. Um forward avalia todas; sub-batching controla a VRAM.

    O hook e registrado sob demanda via context manager e removido ao final —
    nunca empilha, independente de re-execucao de celula.

    Parametros
    ----------
    model, tokenizer : Gemma 2 2B (eval) e tokenizer (padding_side='left').
    sae_encoder : SAEEncoderForward.
    sae_layer : int, camada cujo residual stream a SAE le.
    device : str.
    max_len : int, truncamento (default 256).
    sub_batch : int, quantas ablacoes avaliar por forward (controle de VRAM).
    """

    def __init__(self, model, tokenizer, sae_encoder: SAEEncoderForward,
                 sae_layer: int, device: str, max_len: int = 256,
                 sub_batch: int = 32):
        self.model       = model
        self.tokenizer   = tokenizer
        self.sae_encoder = sae_encoder
        self.sae_layer   = sae_layer
        self.device      = device
        self.max_len     = max_len
        self.sub_batch   = sub_batch

    @contextlib.contextmanager
    def _capture_residual(self):
        captured = {}

        def hook(module, inputs, outputs):
            captured["h"] = outputs[0] if isinstance(outputs, tuple) else outputs

        handle = self.model.model.layers[self.sae_layer].register_forward_hook(hook)
        try:
            yield captured
        finally:
            handle.remove()

    @torch.no_grad()
    def _forward_activation(self, input_ids: torch.Tensor,
                            attn_mask: torch.Tensor, feature_idx: int) -> torch.Tensor:
        """
        Roda o modelo para um batch (B, T) com attention masks (B, T) e retorna
        a ativacao da feature no last-token para cada item: (B,).
        """
        with self._capture_residual() as captured:
            _ = self.model(input_ids=input_ids, attention_mask=attn_mask)
            h = captured["h"]                       # (B, T, d_model)
        h_last = h[:, -1, :].float()                # (B, d_model)
        return self.sae_encoder.feature_activation(h_last, feature_idx)  # (B,)

    def ablate(self, text: str, feature_idx: int,
               skip_bos: bool = True, skip_last: bool = True):
        """
        Ablacao causal por posicao para a feature `feature_idx` no `text`.

        Para cada posicao ablacionavel j, computa a queda de ativacao:
            attr(j) = act_baseline - act_ablacionado(j)
        Valores positivos = token sustenta a feature (remove-lo derruba a ativacao).
        Valores negativos = token suprime a feature (remove-lo aumenta a ativacao).

        Parametros
        ----------
        skip_bos  : nao ablaciona a posicao 0 (BOS/attention sink). Sua ablacao
                    colapsa a redistribuicao de atencao e nao mede semantica.
        skip_last : nao ablaciona o last-token (posicao de leitura da ativacao).

        Retorna
        -------
        tokens : list[str]  tokens decodificados, sem padding
        attr   : np.ndarray (n_tokens,) queda de ativacao por token; NaN nas
                 posicoes puladas (BOS/last), para alinhamento com `tokens`.
        act    : float      ativacao baseline da feature (todos os tokens)
        """
        tok = self.tokenizer

        # Tokeniza (sem padding aqui: um texto so; padding e por replicacao depois)
        enc = tok(text, return_tensors="pt", truncation=True,
                  max_length=self.max_len).to(self.device)
        input_ids = enc["input_ids"]                # (1, T)
        base_mask = enc["attention_mask"]           # (1, T) — normalmente tudo 1
        T = input_ids.shape[1]

        # --- Baseline: ativacao com todos os tokens ---
        act_base = self._forward_activation(input_ids, base_mask, feature_idx).item()

        # --- Decide quais posicoes ablacionar ---
        positions = list(range(T))
        if skip_bos and T > 0:
            # Detecta BOS na posicao 0 (Gemma insere <bos> no inicio)
            if tok.bos_token_id is not None and input_ids[0, 0].item() == tok.bos_token_id:
                positions = [p for p in positions if p != 0]
        if skip_last and T > 0:
            positions = [p for p in positions if p != T - 1]

        # --- Ablacao batched: replica o texto e zera uma posicao por linha ---
        attr = np.full(T, np.nan, dtype=np.float32)

        for start in range(0, len(positions), self.sub_batch):
            chunk = positions[start:start + self.sub_batch]
            B = len(chunk)

            # Replica input_ids B vezes; masks partem do baseline
            ids_batch  = input_ids.expand(B, T).contiguous()          # (B, T)
            mask_batch = base_mask.expand(B, T).contiguous().clone()  # (B, T)

            # Cada linha b zera a atencao para a posicao chunk[b]
            for b, pos in enumerate(chunk):
                mask_batch[b, pos] = 0

            acts = self._forward_activation(ids_batch, mask_batch, feature_idx)  # (B,)
            acts = acts.cpu().numpy()

            # attr = queda em relacao ao baseline
            for b, pos in enumerate(chunk):
                attr[pos] = act_base - acts[b]

        # Decodifica tokens (todas as posicoes, para alinhar com attr)
        ids    = input_ids.squeeze(0).cpu().tolist()
        tokens = [tok.decode([tid]) for tid in ids]

        return tokens, attr, float(act_base)


# =============================================================================
# 3. Carregamento de artefatos (reaproveita metodo2_traceback quando disponivel)
# =============================================================================
# Para manter este modulo autonomo, replicamos os loaders essenciais. Se voce ja
# importou metodo2_traceback, pode usar os de la — sao equivalentes.

def load_pipeline_artifacts(drive_dir: str, sae_layer, sae_width, sae_l0):
    """Carrega o .npz de ativacoes SAE do SAEs.ipynb e deriva alive_indices."""
    fname = f"sae_acts_dbpedia_gemma2b_layer{sae_layer}_width{sae_width}_l0{sae_l0}.npz"
    data  = np.load(os.path.join(drive_dir, fname))

    sae_acts = data["sae_acts"]
    labels   = data["labels"]
    indices  = data["indices"]

    alive_mask    = (sae_acts > 0).any(axis=0)
    alive_indices = np.where(alive_mask)[0]

    label_names = ["Company", "School", "Artist", "Athlete", "Politics",
                   "Transportation", "Building", "Nature", "Village", "Animal",
                   "Plant", "Album", "Film", "WrittenWork"]

    return {
        "sae_acts": sae_acts, "labels": labels, "indices": indices,
        "alive_mask": alive_mask, "alive_indices": alive_indices,
        "label_names": label_names,
        "sae_layer": data["sae_layer"].item(), "sae_width": data["sae_width"].item(),
        "sae_l0": data["sae_l0"].item(), "d_model": data["d_model"].item(),
        "d_sae": data["d_sae"].item(), "N": sae_acts.shape[0],
    }


def load_texts(indices, hf_dataset: str = "fancyzhx/dbpedia_14",
               split: str = "test", text_field: str = "content"):
    """Recupera os textos originais do DBpedia a partir dos indices preservados."""
    from datasets import load_dataset
    ds = load_dataset(hf_dataset, split=split)
    return [ds[int(i)][text_field] for i in indices]


def load_gemma(device: str, model_name: str = "google/gemma-2-2b",
               dtype=torch.bfloat16):
    """Carrega tokenizer e modelo com as convencoes do pipeline (left pad, bf16)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto",
        output_hidden_states=False,
    )
    model.eval()
    return tokenizer, model


def load_feature_catalog(drive_dir, sae_layer, sae_width, sae_l0, enriched=True):
    suffix = "_enriched" if enriched else ""
    fdb_path = os.path.join(
        drive_dir,
        f"feature_interpretations{suffix}_layer{sae_layer}_width{sae_width}_l0{sae_l0}.json"
    )
    if not os.path.exists(fdb_path):
        return {}
    with open(fdb_path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def load_fadex_phis(drive_dir: str, sae_layer, sae_width, sae_l0,
                    n: int = None, n_alive: int = None):
    """Carrega all_phis do FADEx_SAE.ipynb, validando shapes."""
    phis_path = os.path.join(
        drive_dir, f"fadex_phis_layer{sae_layer}_width{sae_width}_l0{sae_l0}.npz"
    )
    all_phis = np.load(phis_path)["all_phis"]
    if n is not None:
        assert all_phis.shape[0] == n, \
            f"all_phis tem {all_phis.shape[0]} linhas, esperado N={n}"
    if n_alive is not None:
        assert all_phis.shape[1] == n_alive, \
            f"all_phis tem {all_phis.shape[1]} colunas, esperado n_alive={n_alive}"
    return all_phis


# =============================================================================
# 4. Selecao de features e catalogo
# =============================================================================

def concept_of(sae_idx: int, feature_db: dict) -> str:
    """Rotulo legivel de uma feature SAE. Prefere Neuronpedia, cai para auto_label."""
    entry = feature_db.get(int(sae_idx))
    if entry is None:
        return "sem catalogo"
    np_label = entry.get("neuronpedia_label")
    if np_label:
        return np_label
    return entry.get("auto_label", "sem catalogo")

def top_fadex_features(text_idx: int, all_phis: np.ndarray,
                       alive_indices: np.ndarray, k: int = 5):
    """
    Top-k features SAE que o FADEx aponta como mais importantes para a posicao
    do documento. Usa |phi| (magnitude). Retorna [(sae_idx, score, col_j), ...].
    """
    phi_row = np.abs(all_phis[text_idx])
    order   = np.argsort(phi_row)[::-1][:k]
    return [(int(alive_indices[j]), float(phi_row[j]), int(j)) for j in order]


def most_active_feature(text_idx: int, sae_acts: np.ndarray,
                        alive_mask: np.ndarray, alive_indices: np.ndarray) -> int:
    """Fallback: feature que ATIVA mais forte no texto (nao passa pelo FADEx)."""
    local = sae_acts[text_idx, alive_mask]
    return int(alive_indices[np.argmax(local)])


# =============================================================================
# 5. Visualizacao
# =============================================================================

def render_token_heatmap(tokens, attr, title: str = "",
                         drop_first: bool = False, drop_last: bool = False,
                         robust: bool = True, signed: bool = True,
                         cmap_name: str = "Reds", cmap_signed: str = "RdBu_r"):
    """
    Renderiza tokens com fundo colorido proporcional a queda de ativacao (ablacao).

    Diferente do Metodo 2, a ablacao produz valores COM SINAL:
      - positivo  = token sustenta a feature (remove-lo derruba a ativacao)
      - negativo  = token suprime a feature (remove-lo aumenta a ativacao)

    Parametros
    ----------
    signed : se True, usa colormap divergente (RdBu_r) centrado em zero — vermelho
             para tokens que sustentam, azul para os que suprimem. Se False, usa
             magnitude absoluta com colormap sequencial.
    drop_first / drop_last : neutraliza BOS / last-token na escala. Como a ablacao
             ja pula essas posicoes por padrao (viram NaN), normalmente nao e
             necessario — os NaN sao tratados como neutros aqui.
    robust : normalizacao por percentil (robusta a outliers).

    Posicoes NaN (puladas na ablacao) sao renderizadas em cinza neutro.
    """
    from matplotlib import cm
    from IPython.display import HTML, display

    a = np.asarray(attr, dtype=float).copy()
    nan_mask = np.isnan(a)

    # Trabalha so com valores validos para definir a escala
    valid = a[~nan_mask]
    if valid.size == 0:
        valid = np.array([0.0])

    if drop_first and len(a) > 1 and not nan_mask[0]:
        a[0] = np.nan; nan_mask[0] = True
    if drop_last and len(a) > 1 and not nan_mask[-1]:
        a[-1] = np.nan; nan_mask[-1] = True
        valid = a[~nan_mask]

    if signed:
        # Escala simetrica centrada em zero
        if robust:
            m = np.percentile(np.abs(valid), 98) + 1e-12
        else:
            m = np.abs(valid).max() + 1e-12
        norm = np.clip((a / m) * 0.5 + 0.5, 0.0, 1.0)   # 0.5 = zero (branco)
        cmap = cm.get_cmap(cmap_signed)
    else:
        b = np.abs(a)
        if robust:
            lo, hi = np.percentile(b[~nan_mask], 2), np.percentile(b[~nan_mask], 98)
        else:
            lo, hi = np.nanmin(b), np.nanmax(b)
        norm = np.clip((b - lo) / (hi - lo + 1e-12), 0.0, 1.0)
        cmap = cm.get_cmap(cmap_name)

    spans = []
    for tok, val, is_nan in zip(tokens, norm, nan_mask):
        if is_nan:
            rgb = "rgb(230,230,230)"   # cinza neutro para posicoes puladas
        else:
            r, g, b_, _ = cmap(float(val))
            rgb = f"rgb({int(r*255)},{int(g*255)},{int(b_*255)})"
        safe = tok.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        spans.append(
            f'<span style="background-color:{rgb}; color:black; padding:1px 2px; '
            f'border-radius:3px; margin:1px;">{safe}</span>'
        )
    html = "<div style='font-family:monospace; line-height:2.2; font-size:14px;'>"
    if title:
        html += f"<b>{title}</b><br>"
    if signed:
        html += ("<span style='font-size:11px; color:#666;'>"
                 "vermelho = sustenta a feature · azul = suprime · cinza = nao ablacionado"
                 "</span><br>")
    html += "".join(spans) + "</div>"
    display(HTML(html))


# =============================================================================
# 6. Relatorio de alto nivel
# =============================================================================

def report_text_ablation(text_idx: int, texts, artifacts: dict,
                         all_phis: np.ndarray, feature_db: dict,
                         engine: AblationEngine, top_k: int = 5,
                         skip_bos: bool = True, skip_last: bool = True,
                         heatmap_kwargs: dict = None):
    """
    Relatorio completo de ablacao para um documento:
      1. imprime o texto e categoria
      2. lista as top-k features do FADEx (score FADEx, ativacao SAE, conceito)
      3. renderiza um heatmap de ablacao causal por feature

    Fecha o loop: FADEx seleciona QUAIS features explicam a posicao; a ablacao
    mostra QUAIS tokens causalmente sustentam cada feature (com sinal).
    """
    heatmap_kwargs = heatmap_kwargs or {}
    labels        = artifacts["labels"]
    label_names   = artifacts["label_names"]
    sae_acts      = artifacts["sae_acts"]
    alive_indices = artifacts["alive_indices"]

    print("=" * 80)
    print(f"TEXTO #{text_idx}  -  categoria: {label_names[labels[text_idx]]}")
    print("=" * 80)
    print(texts[text_idx])
    print()

    feats = top_fadex_features(text_idx, all_phis, alive_indices, k=top_k)
    print(f"TOP {top_k} FEATURES MAIS IMPORTANTES (FADEx) PARA A POSICAO DESTE TEXTO")
    print("-" * 80)
    for rank, (sae_idx, fadex_score, _) in enumerate(feats, 1):
        activation = float(sae_acts[text_idx, sae_idx])
        concept    = concept_of(sae_idx, feature_db)
        print(f"{rank:2d}. SAE #{sae_idx:<6d}  FADEx|phi|={fadex_score:.4f}  "
              f"act={activation:.3f}  [{concept}]")
    print()

    print("HEATMAPS (Ablacao causal) - quais tokens sustentam cada feature")
    print("-" * 80)
    for rank, (sae_idx, _, _) in enumerate(feats, 1):
        tokens, attr, act = engine.ablate(
            texts[text_idx], sae_idx, skip_bos=skip_bos, skip_last=skip_last
        )
        render_token_heatmap(
            tokens, attr,
            title=f"#{rank} · Feature SAE {sae_idx} "
                  f"({concept_of(sae_idx, feature_db)}) · ativacao baseline={act:.3f}",
            **heatmap_kwargs
        )


# =============================================================================
# 7. Comparacao Metodo 2 (Grad x Input) vs Metodo 3 (Ablacao) — validacao
# =============================================================================

def compare_attribution(text, feature_idx, ablation_engine: AblationEngine,
                        gradxinput_fn, skip_bos: bool = True, skip_last: bool = True):
    """
    Compara lado a lado a atribuicao por Gradient x Input (Metodo 2) e por Ablacao
    (Metodo 3) para a mesma feature no mesmo texto. Util para validar a fidelidade
    do Metodo 2: onde ambos concordam, ganha-se confianca no metodo barato; onde
    divergem, ha uma feature interessante (saturacao de gradiente, efeito nao-linear).

    Parametros
    ----------
    gradxinput_fn : callable(text, feature_idx) -> (tokens, attr, act)
        Tipicamente engine_m2.traceback do modulo metodo2_traceback.

    Retorna dict com tokens, attr_grad, attr_ablation, e a correlacao de Spearman
    entre as duas atribuicoes (nas posicoes validas de ambas).
    """
    from scipy.stats import spearmanr

    tokens_g, attr_g, act_g = gradxinput_fn(text, feature_idx)
    tokens_a, attr_a, act_a = ablation_engine.ablate(
        text, feature_idx, skip_bos=skip_bos, skip_last=skip_last
    )

    # Alinhamento: ambos devem produzir a mesma tokenizacao (mesmo tokenizer,
    # mesmo max_len). Grad x Input remove padding; ablacao mantem todas as
    # posicoes. Assumimos textos sem padding (batch de 1), entao alinham.
    n = min(len(attr_g), len(attr_a))
    ag = np.asarray(attr_g[:n], dtype=float)
    aa = np.asarray(attr_a[:n], dtype=float)

    # Correlacao so nas posicoes onde ambos sao validos (ablacao tem NaN nos pulados)
    valid = ~np.isnan(aa)
    if valid.sum() >= 3:
        rho, pval = spearmanr(ag[valid], aa[valid])
    else:
        rho, pval = np.nan, np.nan

    return {
        "tokens":        tokens_g[:n],
        "attr_grad":     ag,
        "attr_ablation": aa,
        "act":           act_g,
        "spearman_rho":  float(rho),
        "spearman_p":    float(pval),
    }