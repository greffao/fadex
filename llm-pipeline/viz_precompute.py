# -*- coding: utf-8 -*-
"""
viz_precompute.py

Fase A da interface de visualizacao: pre-computa, para TODOS os N pontos, as
top-k features FADEx e o traceback (Metodo 2 = Gradient x Input, ou Metodo 3 =
Ablacao) de cada feature, ja renderizado como lista de (token, score). O
resultado e salvo em JSON no Drive e consumido pela interface (viz_interface.py),
que roda sem o Gemma carregado.

Esta separacao e o que torna a interface fluida: todo o custo do forward/backward
do Gemma e pago aqui, uma vez, offline. A interface so le dados.

Requer, ja carregados no ambiente (via os notebooks Metodo_2 / Metodo_3):
    - metodo2_traceback.TracebackEngine  (se METHOD='gradxinput')
    - metodo3_ablacao.AblationEngine     (se METHOD='ablation')
    - artefatos do pipeline (sae_acts, labels, alive_indices, textos)
    - all_phis do FADEx

Uso tipico (no notebook de pre-computacao):

    import viz_precompute as vp
    vp.precompute_all(
        out_path   = os.path.join(DRIVE_DIR, "viz_traceback_layer12_16k_ablation.json"),
        texts      = texts,
        artifacts  = art,
        all_phis   = all_phis,
        all_norms  = all_norms,
        feature_db = feature_db,
        engine     = ablation_engine,      # ou traceback_engine
        method     = "ablation",           # ou "gradxinput"
        top_k      = 5,
    )
"""

import os
import json
import time

import numpy as np
from tqdm.auto import tqdm


def _traceback_one(engine, method, text, sae_idx, skip_bos=True, skip_last=True):
    """
    Roda o traceback de uma feature em um texto, normalizado para o formato comum
    (tokens, scores, activation). Abstrai a diferenca de assinatura entre os dois
    motores.

    Retorna (tokens: list[str], scores: list[float], activation: float).
    Scores podem conter NaN (ablacao, posicoes puladas) — serializados como None.
    """
    if method == "gradxinput":
        tokens, attr, act = engine.traceback(text, sae_idx)
    elif method == "ablation":
        tokens, attr, act = engine.ablate(text, sae_idx,
                                          skip_bos=skip_bos, skip_last=skip_last)
    else:
        raise ValueError(f"method desconhecido: {method}")

    # NaN -> None para JSON valido
    scores = [None if (v is None or np.isnan(v)) else float(v) for v in attr]
    return tokens, scores, float(act)


def _concept_of(sae_idx, feature_db):
    """Rotulo legivel, preferindo Neuronpedia se disponivel."""
    entry = feature_db.get(int(sae_idx))
    if entry is None:
        return "sem catalogo"
    np_label = entry.get("neuronpedia_label")
    if np_label:
        return np_label
    return entry.get("auto_label", "sem catalogo")


def precompute_all(out_path, texts, artifacts, all_phis, all_norms, feature_db,
                   engine, method="ablation", top_k=5,
                   skip_bos=True, skip_last=True,
                   save_every=200, resume=True):
    """
    Pre-computa o payload completo da interface para todos os pontos.

    Estrutura do JSON de saida:
    {
      "meta": {method, top_k, n_points, sae_layer, ...},
      "points": [
        {
          "idx": int,
          "label": int,
          "label_name": str,
          "snd": float,                     # distorcao local (all_norms)
          "x": float, "y": float,           # coords 2D (preenchidas na interface)
          "features": [
            {
              "sae_idx": int,
              "concept": str,
              "fadex_score": float,
              "activation": float,
              "tokens": [str, ...],
              "scores": [float|None, ...],  # alinhado com tokens
            }, ... (top_k)
          ]
        }, ... (N pontos)
      ]
    }

    Parametros
    ----------
    save_every : salva parcial a cada X pontos (resiliencia a timeout do Colab).
    resume     : se True e out_path existe, retoma de onde parou.
    """
    labels        = artifacts["labels"]
    label_names   = artifacts["label_names"]
    sae_acts      = artifacts["sae_acts"]
    alive_indices = artifacts["alive_indices"]
    N             = artifacts["N"]

    # --- Retomada: carrega parcial se existir ---
    points = [None] * N
    done   = set()
    if resume and os.path.exists(out_path):
        with open(out_path) as f:
            prev = json.load(f)
        for p in prev.get("points", []):
            if p is not None:
                points[p["idx"]] = p
                done.add(p["idx"])
        print(f"Retomando: {len(done)}/{N} pontos ja computados.")

    meta = {
        "method":    method,
        "top_k":     top_k,
        "n_points":  N,
        "sae_layer": artifacts["sae_layer"],
        "sae_width": artifacts["sae_width"],
        "sae_l0":    artifacts["sae_l0"],
        "skip_bos":  skip_bos,
        "skip_last": skip_last,
    }

    def _flush():
        with open(out_path, "w") as f:
            json.dump({"meta": meta, "points": points}, f, ensure_ascii=False)

    t0 = time.perf_counter()
    processed = 0

    for i in tqdm(range(N), desc=f"Traceback ({method})"):
        if i in done:
            continue

        # top-k features FADEx para este ponto (por |phi|)
        phi_row = np.abs(all_phis[i])
        order   = np.argsort(phi_row)[::-1][:top_k]

        feats = []
        for j in order:
            sae_idx = int(alive_indices[j])
            tokens, scores, act = _traceback_one(
                engine, method, texts[i], sae_idx,
                skip_bos=skip_bos, skip_last=skip_last
            )
            feats.append({
                "sae_idx":     sae_idx,
                "concept":     _concept_of(sae_idx, feature_db),
                "fadex_score": float(phi_row[j]),
                "activation":  act,
                "tokens":      tokens,
                "scores":      scores,
            })

        points[i] = {
            "idx":        i,
            "label":      int(labels[i]),
            "label_name": label_names[int(labels[i])],
            "snd":        float(all_norms[i]),
            "x":          None,   # preenchido na interface a partir da projecao
            "y":          None,
            "features":   feats,
        }
        processed += 1

        if processed % save_every == 0:
            _flush()

    _flush()
    dt = time.perf_counter() - t0
    print(f"\nConcluido: {N} pontos em {dt:.1f}s. Salvo em {out_path}")
    return out_path
