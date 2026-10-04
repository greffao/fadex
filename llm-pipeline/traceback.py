# -*- coding: utf-8 -*-
"""
metodo2_traceback.py

Modulo de traceback token-level para features SAE via Gradient x Input (Metodo 2).

Dada uma feature SAE identificada pelo FADEx como importante para a posicao de um
documento na projecao 2D, este modulo responde: quais tokens do texto ativam essa
feature? A atribuicao propaga o gradiente da ativacao da feature (no last-token) de
volta ate os embeddings de entrada, capturando o efeito da atencao causal.

Pipeline de referencia:
    SAEs.ipynb      -> extrai sae_acts (ativacoes SAE por documento, last-token)
    FADEx_SAE.ipynb -> DR + FADEx, produz all_phis (atribuicao por feature)
    Metodo_2.ipynb  -> (este modulo) traceback das features FADEx para tokens

Convencoes herdadas do pipeline:
    - left padding + last-token pooling (posicao -1 = ultimo token de conteudo)
    - encoder SAE em fp32 (sensibilidade do threshold JumpReLU)
    - Gemma 2 2B em bf16

Uso tipico (no notebook orquestrador):

    from metodo2_traceback import (
        load_pipeline_artifacts, load_texts, load_gemma, load_sae_encoder,
        load_feature_catalog, load_fadex_phis, TracebackEngine,
        render_token_heatmap, report_text_traceback,
    )

    art   = load_pipeline_artifacts(DRIVE_DIR, SAE_LAYER, SAE_WIDTH, SAE_L0)
    texts = load_texts(art["indices"])
    tok, model = load_gemma(device)
    enc   = load_sae_encoder(SAE_REPO, SAE_PATH, art["d_model"], art["d_sae"], device)
    fdb   = load_feature_catalog(DRIVE_DIR, SAE_LAYER, SAE_WIDTH, SAE_L0)
    phis  = load_fadex_phis(DRIVE_DIR, SAE_LAYER, SAE_WIDTH, SAE_L0,
                            n=art["N"], n_alive=len(art["alive_indices"]))

    engine = TracebackEngine(model, tok, enc, art["sae_layer"], device)
    report_text_traceback(TEXT_IDX, texts, art, phis, fdb, engine, top_k=5)
"""

import os
import json
import contextlib

import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# 1. Encoder SAE diferenciavel
# =============================================================================

class DifferentiableSAEEncoder:
    """
    Wrapper do encoder JumpReLU que permite gradientes fluirem atraves de x.

    Diferente da JumpReLUSAE do notebook de extracao (que usa @torch.no_grad()),
    aqui os tensores nao sao envoltos em no_grad: o objetivo e justamente que o
    autograd conecte a ativacao da feature de volta aos embeddings de entrada.
    Os pesos sao constantes (nao ha treino), mas o grafo passa por eles ate x.

    Parametros
    ----------
    W_enc, b_enc, theta : torch.Tensor
        Pesos do encoder ja no device correto, em fp32.
    """

    def __init__(self, W_enc: torch.Tensor, b_enc: torch.Tensor, theta: torch.Tensor):
        self.W_enc = W_enc          # (d_model, d_sae)
        self.b_enc = b_enc          # (d_sae,)
        self.theta = theta          # (d_sae,)

    def preactivation(self, x: torch.Tensor) -> torch.Tensor:
        """pre = x W_enc + b_enc. x: (..., d_model) fp32. Diferenciavel em x."""
        return x @ self.W_enc + self.b_enc

    def feature_activation(self, x: torch.Tensor, feature_idx: int) -> torch.Tensor:
        """
        Ativacao JumpReLU da feature: pre_i * 1[pre_i > theta_i].

        A mascara 1[pre>theta] e localmente constante (gradiente zero q.t.p.).
        Ao multiplica-la por pre_i, o autograd propaga somente pelo ramo pre_i,
        que e exatamente a saliencia desejada: quanto o valor que passou pelo
        gate depende do input. Se a feature estiver inativa (gate=0), o alvo e
        zero e o traceback nao faz sentido (a feature nao ativou nesse texto).
        """
        pre   = self.preactivation(x)
        pre_i = pre[..., feature_idx]
        gate  = (pre_i > self.theta[feature_idx]).float()   # 0/1, sem grad
        return pre_i * gate


def load_sae_encoder(sae_repo: str, sae_path: str, d_model: int, d_sae: int,
                     device: str):
    """
    Baixa os pesos da SAE Gemma Scope e monta o encoder diferenciavel.

    So carrega W_enc, b_enc e threshold (o decoder nao e necessario para
    traceback). Tudo em fp32 pela sensibilidade do threshold JumpReLU.
    """
    from huggingface_hub import hf_hub_download

    sae_file = hf_hub_download(repo_id=sae_repo, filename=sae_path)
    params   = np.load(sae_file)

    W_enc = torch.from_numpy(params["W_enc"]).float().to(device)      # (d_model, d_sae)
    b_enc = torch.from_numpy(params["b_enc"]).float().to(device)      # (d_sae,)
    theta = torch.from_numpy(params["threshold"]).float().to(device)  # (d_sae,)

    assert W_enc.shape == (d_model, d_sae), \
        f"W_enc shape {tuple(W_enc.shape)} != esperado ({d_model}, {d_sae})"

    return DifferentiableSAEEncoder(W_enc, b_enc, theta)


# =============================================================================
# 2. Motor de traceback (Gradient x Input)
# =============================================================================

class TracebackEngine:
    """
    Executa o traceback Gradient x Input de uma feature SAE em um texto.

    O hook e registrado sob demanda dentro de cada chamada, via context manager,
    e removido ao final — nunca empilha, independente de quantas vezes a celula
    do notebook seja re-executada. Isso elimina a classe de bug de "backward
    through the graph a second time" causada por hooks duplicados.

    Parametros
    ----------
    model : transformers CausalLM (Gemma 2 2B), em modo eval.
    tokenizer : tokenizer correspondente, com padding_side='left'.
    sae_encoder : DifferentiableSAEEncoder.
    sae_layer : int, indice da camada cujo residual stream a SAE le.
    device : str.
    max_len : int, truncamento de tokens (default 256, igual a extracao).
    """

    def __init__(self, model, tokenizer, sae_encoder: DifferentiableSAEEncoder,
                 sae_layer: int, device: str, max_len: int = 256):
        self.model       = model
        self.tokenizer   = tokenizer
        self.sae_encoder = sae_encoder
        self.sae_layer   = sae_layer
        self.device      = device
        self.max_len     = max_len

    @contextlib.contextmanager
    def _capture_residual(self):
        """
        Context manager que registra um hook temporario na camada SAE e captura
        o residual stream de saida. Remove o hook ao sair, sempre (mesmo em erro).
        """
        captured = {}

        def hook(module, inputs, outputs):
            captured["h"] = outputs[0] if isinstance(outputs, tuple) else outputs

        handle = self.model.model.layers[self.sae_layer].register_forward_hook(hook)
        try:
            yield captured
        finally:
            handle.remove()

    def traceback(self, text: str, feature_idx: int):
        """
        Computa atribuicao Gradient x Input de cada token para a ativacao da
        feature SAE `feature_idx`, medida no last-token.

        Retorna
        -------
        tokens : list[str]  tokens decodificados, sem padding
        attr   : np.ndarray (n_tokens,) saliencia por token (norma L2 de g*x)
        act    : float      valor da ativacao da feature (referencia)
        """
        tok = self.tokenizer

        # 1) Tokeniza (mesma convencao da extracao: left pad, truncation)
        enc = tok(text, return_tensors="pt", truncation=True,
                  max_length=self.max_len).to(self.device)
        input_ids = enc["input_ids"]                 # (1, T)
        attn_mask = enc["attention_mask"]            # (1, T)

        # 2) Embeddings como folha diferenciavel
        embed_layer   = self.model.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids).clone().detach()   # (1, T, d_model) bf16
        inputs_embeds.requires_grad_(True)

        # 3) Forward via inputs_embeds, com hook temporario capturando o residual
        self.model.zero_grad(set_to_none=True)
        with self._capture_residual() as captured:
            _ = self.model(inputs_embeds=inputs_embeds, attention_mask=attn_mask)
            h = captured["h"]                        # (1, T, d_model)

        # 4) Residual do last-token em fp32 (SAE espera fp32)
        h_last = h[:, -1, :].float()                 # (1, d_model)

        # 5) Alvo escalar: ativacao da feature
        y = self.sae_encoder.feature_activation(h_last, feature_idx).squeeze()

        # 6) Backward: dy/d(inputs_embeds)
        y.backward()
        grads = inputs_embeds.grad                   # (1, T, d_model)

        # 7) Gradient x Input -> norma L2 por token
        gxi  = grads.float() * inputs_embeds.float()  # (1, T, d_model)
        attr = gxi.norm(dim=-1).squeeze(0)            # (T,)

        # 8) Remove padding (left pad -> mask indica validos)
        valid  = attn_mask.squeeze(0).bool()
        attr   = attr[valid].detach().cpu().numpy()
        ids    = input_ids.squeeze(0)[valid].cpu().tolist()
        tokens = [tok.decode([tid]) for tid in ids]

        return tokens, attr, float(y.detach().cpu())


# =============================================================================
# 3. Carregamento de artefatos do pipeline
# =============================================================================

def load_pipeline_artifacts(drive_dir: str, sae_layer, sae_width, sae_l0):
    """
    Carrega o .npz de ativacoes SAE salvo pelo SAEs.ipynb e deriva o mapeamento
    de features alive.

    Retorna dict com: sae_acts, labels, indices, alive_mask, alive_indices,
    label_names, sae_layer, sae_width, sae_l0, d_model, d_sae, N.
    """
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
        "sae_acts":      sae_acts,
        "labels":        labels,
        "indices":       indices,
        "alive_mask":    alive_mask,
        "alive_indices": alive_indices,
        "label_names":   label_names,
        "sae_layer":     data["sae_layer"].item(),
        "sae_width":     data["sae_width"].item(),
        "sae_l0":        data["sae_l0"].item(),
        "d_model":       data["d_model"].item(),
        "d_sae":         data["d_sae"].item(),
        "N":             sae_acts.shape[0],
    }


def load_texts(indices, hf_dataset: str = "fancyzhx/dbpedia_14",
               split: str = "test", text_field: str = "content"):
    """Recupera os textos originais do DBpedia a partir dos indices preservados."""
    from datasets import load_dataset
    ds = load_dataset(hf_dataset, split=split)
    texts = [ds[int(i)][text_field] for i in indices]
    return texts


def load_gemma(device: str, model_name: str = "google/gemma-2-2b",
               dtype=torch.bfloat16):
    """
    Carrega tokenizer e modelo com as convencoes do pipeline:
    left padding (last-token pooling) e bf16.
    """
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
    """
    Carrega all_phis (atribuicoes do FADEx) salvo pelo FADEx_SAE.ipynb.

    Valida shapes contra n e n_alive quando fornecidos.
    Retorna np.ndarray (N, n_alive).
    """
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
    """Rotulo legivel de uma feature SAE a partir do catalogo (fallback: sem catalogo)."""
    entry = feature_db.get(int(sae_idx))
    if entry is None:
        return "sem catalogo"
    return entry.get("auto_label", "sem catalogo")


def top_fadex_features(text_idx: int, all_phis: np.ndarray,
                       alive_indices: np.ndarray, k: int = 5):
    """
    Retorna as top-k features SAE que o FADEx aponta como mais importantes para
    a posicao do documento `text_idx` na projecao.

    Usa |phi| (magnitude): a atribuicao tem sinal, mas importancia e magnitude,
    consistente com o ranqueamento global do FADEx_SAE.ipynb.

    Retorna lista de tuplas (sae_idx, fadex_score, column_j).
    """
    phi_row = np.abs(all_phis[text_idx])              # (n_alive,)
    order   = np.argsort(phi_row)[::-1][:k]           # colunas de X
    return [(int(alive_indices[j]), float(phi_row[j]), int(j)) for j in order]


def most_active_feature(text_idx: int, sae_acts: np.ndarray,
                        alive_mask: np.ndarray, alive_indices: np.ndarray) -> int:
    """
    Fallback: feature SAE que ATIVA mais forte no texto (nao passa pelo FADEx).
    Util quando all_phis ainda nao foi computado. Retorna indice SAE.
    """
    local = sae_acts[text_idx, alive_mask]
    return int(alive_indices[np.argmax(local)])


# =============================================================================
# 5. Visualizacao
# =============================================================================

def render_token_heatmap(tokens, attr, title: str = "",
                         drop_first: bool = True, drop_last: bool = True,
                         robust: bool = True, cmap_name: str = "Reds"):
    """
    Renderiza tokens com fundo colorido proporcional a atribuicao (HTML inline).

    Parametros de controle de artefatos estruturais:
      drop_first : neutraliza o token BOS na escala de cor (attention sink posicional)
      drop_last  : neutraliza o ultimo token (dominado pela skip connection residual)
      robust     : normalizacao por percentil [2, 98] em vez de min-max — robusta
                   a outliers residuais (sinks nao mascarados, tokens raros)

    Nota metodologica: drop_first/drop_last mascaram APENAS os dois tokens cujo
    papel estrutural (nao-semantico) e conhecido e posicional. Nao se filtra
    pontuacao por identidade de token, pois pontuacao pode carregar sinal
    semantico legitimo (ex.: "(2003)" num album, virgulas em datas biograficas).
    """
    from matplotlib import cm
    from IPython.display import HTML, display

    a = np.asarray(attr, dtype=float).copy()

    if drop_first and len(a) > 1:
        a[0] = a[1:].min()
    if drop_last and len(a) > 1:
        a[-1] = a[:-1].min()

    if robust:
        lo, hi = np.percentile(a, 2), np.percentile(a, 98)
        a = np.clip((a - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    else:
        lo, hi = a.min(), a.max()
        a = (a - lo) / (hi - lo + 1e-12)

    cmap = cm.get_cmap(cmap_name)
    spans = []
    for tok, val in zip(tokens, a):
        r, g, b, _ = cmap(float(val))
        rgb = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
        safe = tok.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        spans.append(
            f'<span style="background-color:{rgb}; color:black; padding:1px 2px; '
            f'border-radius:3px; margin:1px;">{safe}</span>'
        )
    html = "<div style='font-family:monospace; line-height:2.2; font-size:14px;'>"
    if title:
        html += f"<b>{title}</b><br>"
    html += "".join(spans) + "</div>"
    display(HTML(html))


# =============================================================================
# 6. Relatorio de alto nivel (orquestra tudo)
# =============================================================================

def report_text_traceback(text_idx: int, texts, artifacts: dict,
                          all_phis: np.ndarray, feature_db: dict,
                          engine: TracebackEngine, top_k: int = 5,
                          heatmap_kwargs: dict = None):
    """
    Relatorio completo para um documento:
      1. imprime o texto e sua categoria
      2. lista as top-k features do FADEx (com score FADEx, ativacao SAE, conceito)
      3. renderiza um heatmap Gradient x Input por feature

    Fecha o loop conceitual do pipeline: FADEx seleciona QUAIS features explicam
    a posicao do documento; o Metodo 2 mostra QUAIS tokens ativam cada feature.
    """
    heatmap_kwargs = heatmap_kwargs or {}
    labels        = artifacts["labels"]
    label_names   = artifacts["label_names"]
    sae_acts      = artifacts["sae_acts"]
    alive_indices = artifacts["alive_indices"]

    # 1) Texto
    print("=" * 80)
    print(f"TEXTO #{text_idx}  -  categoria: {label_names[labels[text_idx]]}")
    print("=" * 80)
    print(texts[text_idx])
    print()

    # 2) Top-k features FADEx
    feats = top_fadex_features(text_idx, all_phis, alive_indices, k=top_k)
    print(f"TOP {top_k} FEATURES MAIS IMPORTANTES (FADEx) PARA A POSICAO DESTE TEXTO")
    print("-" * 80)
    for rank, (sae_idx, fadex_score, _) in enumerate(feats, 1):
        activation = float(sae_acts[text_idx, sae_idx])
        concept    = concept_of(sae_idx, feature_db)
        print(f"{rank:2d}. SAE #{sae_idx:<6d}  FADEx|phi|={fadex_score:.4f}  "
              f"act={activation:.3f}  [{concept}]")
    print()

    # 3) Heatmaps
    print("HEATMAPS (Gradient x Input) - quais tokens ativam cada feature")
    print("-" * 80)
    for rank, (sae_idx, _, _) in enumerate(feats, 1):
        tokens, attr, act = engine.traceback(texts[text_idx], sae_idx)
        render_token_heatmap(
            tokens, attr,
            title=f"#{rank} · Feature SAE {sae_idx} "
                  f"({concept_of(sae_idx, feature_db)}) · ativacao={act:.3f}",
            **heatmap_kwargs
        )