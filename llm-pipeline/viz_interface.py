# -*- coding: utf-8 -*-
"""
viz_interface.py

Fase B da interface de visualizacao: monta a interface de duas telas em Plotly,
rodando inline no Colab. NAO precisa do Gemma — le apenas dados pre-computados
(viz_precompute.py) e a projecao 2D salva pelo FADEx_SAE.

Tela 1 (esquerda): scatter interativo dos pontos da projecao DR.
    - Alterna entre colorir por CLASSE ou por DISTORCAO (SND), via botao.
    - SND usa colormap divergente (Spectral_r): SND=1 no meio, SND<1 frio,
      SND>1 quente. Implementado com escala divergente centrada em 1.
Tela 2 (direita): ao passar o mouse por um ponto, exibe classe, SND e as top-5
    features FADEx; para cada feature, nome, importancia e o texto original
    pintado (heatmap de traceback do metodo escolhido na pre-computacao).

Como o hover do Plotly nao dispara Python arbitrario inline de forma trivial no
Colab, usamos FigureWidget + callback on_hover, que atualiza um painel HTML
(ipywidgets) a direita. Isso mantem tudo dentro do notebook, sem servidor.
"""

import json

import numpy as np


# =============================================================================
# 1. Carregamento e merge com a projecao
# =============================================================================

def load_payload(traceback_json_path, proj_npz_path):
    """
    Carrega o payload pre-computado (viz_precompute) e injeta as coordenadas 2D
    da projecao salva pelo FADEx_SAE (proj2d_*.npz, chave 'Y2').

    Retorna (meta, points, Y2).
    """
    with open(traceback_json_path) as f:
        data = json.load(f)
    meta   = data["meta"]
    points = data["points"]

    Y2 = np.load(proj_npz_path)["Y2"]
    assert Y2.shape[0] == len(points), \
        f"Projecao tem {Y2.shape[0]} pontos, payload tem {len(points)}"

    # Injeta coords em cada ponto
    for i, p in enumerate(points):
        if p is not None:
            p["x"] = float(Y2[i, 0])
            p["y"] = float(Y2[i, 1])

    return meta, points, Y2


# =============================================================================
# 2. Escala de cor divergente centrada em SND=1
# =============================================================================

def snd_to_color_values(snd_array, center=1.0, clip_percentile=98):
    """
    Mapeia SND para uma escala divergente centrada em `center` (=1).

    Retorna (values_normalizados_0a1, cmin, cmax, cmid) para uso no colorbar.
    A normalizacao e simetrica em torno de `center`: o desvio maximo (robusto,
    por percentil) define os dois extremos, garantindo que center caia no meio
    do colormap divergente.
    """
    snd = np.asarray(snd_array, dtype=float)
    # desvio maximo robusto em relacao ao centro
    dev = np.abs(snd - center)
    max_dev = np.percentile(dev, clip_percentile) + 1e-12
    cmin = center - max_dev
    cmax = center + max_dev
    return snd, cmin, cmax, center


# =============================================================================
# 3. Render de heatmap de tokens como HTML (para o painel direito)
# =============================================================================

def _tokens_to_html(tokens, scores, method, robust=True):
    """
    Converte (tokens, scores) em HTML colorido.

    - gradxinput: scores >= 0, colormap sequencial (Reds).
    - ablation: scores com sinal, colormap divergente (RdBu_r), None = cinza.
    """
    import matplotlib
    from matplotlib import cm

    vals = np.array([np.nan if s is None else s for s in scores], dtype=float)
    valid = vals[~np.isnan(vals)]
    if valid.size == 0:
        valid = np.array([0.0])

    if method == "ablation":
        m = (np.percentile(np.abs(valid), 98) if robust else np.abs(valid).max()) + 1e-12
        norm = np.clip((vals / m) * 0.5 + 0.5, 0.0, 1.0)
        cmap = cm.get_cmap("RdBu_r")
    else:
        lo = np.percentile(valid, 2) if robust else valid.min()
        hi = np.percentile(valid, 98) if robust else valid.max()
        norm = np.clip((vals - lo) / (hi - lo + 1e-12), 0.0, 1.0)
        cmap = cm.get_cmap("Reds")

    spans = []
    for tok, v, is_nan in zip(tokens, norm, np.isnan(vals)):
        if is_nan:
            rgb = "rgb(230,230,230)"
        else:
            r, g, b, _ = cmap(float(v))
            rgb = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
        safe = (tok.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        spans.append(
            f'<span style="background-color:{rgb}; color:black; padding:1px 2px; '
            f'border-radius:3px; margin:1px; display:inline-block;">{safe}</span>'
        )
    return "".join(spans)


def _point_to_html(point, method):
    """Monta o HTML completo do painel direito para um ponto."""
    if point is None:
        return "<i>ponto sem dados</i>"

    legend = ""
    if method == "ablation":
        legend = ("<div style='font-size:11px;color:#666;margin-bottom:6px;'>"
                  "vermelho = sustenta · azul = suprime · cinza = nao ablacionado</div>")

    html = [
        f"<div style='font-family:sans-serif;'>",
        f"<h3 style='margin:0 0 4px 0;'>Ponto #{point['idx']}</h3>",
        f"<div style='font-size:14px;margin-bottom:2px;'>"
        f"<b>Classe:</b> {point['label_name']}</div>",
        f"<div style='font-size:14px;margin-bottom:10px;'>"
        f"<b>Distorcao (SND):</b> {point['snd']:.3f}</div>",
        f"<hr style='border:none;border-top:1px solid #ddd;'>",
        f"<div style='font-size:13px;font-weight:bold;margin:8px 0;'>"
        f"Top-{len(point['features'])} features (FADEx)</div>",
        legend,
    ]

    for rank, feat in enumerate(point["features"], 1):
        heat = _tokens_to_html(feat["tokens"], feat["scores"], method)
        html.append(
            f"<div style='margin-bottom:14px; padding:8px; "
            f"border:1px solid #eee; border-radius:6px;'>"
            f"<div style='font-size:12px;margin-bottom:4px;'>"
            f"<b>#{rank} · SAE {feat['sae_idx']}</b> "
            f"<span style='color:#555;'>({feat['concept']})</span></div>"
            f"<div style='font-size:11px;color:#777;margin-bottom:6px;'>"
            f"FADEx |phi|={feat['fadex_score']:.4f} · "
            f"ativacao={feat['activation']:.3f}</div>"
            f"<div style='font-family:monospace;line-height:2.0;font-size:13px;'>"
            f"{heat}</div>"
            f"</div>"
        )

    html.append("</div>")
    return "".join(html)


# =============================================================================
# 4. Construcao da interface (FigureWidget + painel HTML)
# =============================================================================

def build_interface(meta, points, label_names, label_colors=None,
                    point_size=6, height=750):
    """
    Monta e exibe a interface de duas telas no notebook.

    Esquerda: FigureWidget scatter com botao alternando cor (classe / SND).
    Direita:  painel HTML (ipywidgets.HTML) atualizado no hover.

    Parametros
    ----------
    meta, points : saida de load_payload (com x/y ja injetados).
    label_names  : lista de nomes de classe.
    label_colors : lista de cores por classe (hex). Se None, usa uma paleta.
    """
    import plotly.graph_objects as go
    import ipywidgets as widgets
    from IPython.display import display

    method = meta["method"]

    # Filtra pontos validos e extrai arrays
    valid_pts = [p for p in points if p is not None]
    xs   = np.array([p["x"] for p in valid_pts])
    ys   = np.array([p["y"] for p in valid_pts])
    labs = np.array([p["label"] for p in valid_pts])
    snds = np.array([p["snd"] for p in valid_pts])
    idxs = [p["idx"] for p in valid_pts]

    # --- paleta de classe ---
    if label_colors is None:
        import matplotlib
        from matplotlib import cm
        cmap = cm.get_cmap("tab20")
        label_colors = [matplotlib.colors.to_hex(cmap(k % 20))
                        for k in range(len(label_names))]
    class_colors = [label_colors[l] for l in labs]

    # --- valores de cor para SND (divergente centrado em 1) ---
    snd_vals, cmin, cmax, cmid = snd_to_color_values(snds, center=1.0)

    # --- trace inicial: por CLASSE ---
    scatter = go.Scattergl(
        x=xs, y=ys, mode="markers",
        marker=dict(size=point_size, color=class_colors, line=dict(width=0)),
        customdata=idxs,
        hoverinfo="none",       # o hover e tratado pelo callback, nao pelo tooltip
        showlegend=False,
    )

    fig = go.FigureWidget(data=[scatter])
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(showticklabels=False, zeroline=False, showgrid=False),
        yaxis=dict(showticklabels=False, zeroline=False, showgrid=False),
        title="Projecao (cor: classe)",
        plot_bgcolor="white",
    )

    # --- painel direito ---
    right_panel = widgets.HTML(
        value="<i style='color:#999;'>Passe o mouse sobre um ponto...</i>",
        layout=widgets.Layout(width="100%", height=f"{height}px",
                              overflow_y="auto", padding="8px",
                              border="1px solid #ddd")
    )

    # mapa idx -> point (para lookup rapido no hover)
    idx_to_point = {p["idx"]: p for p in valid_pts}

    def on_hover(trace, ppoints, state):
        if not ppoints.point_inds:
            return
        k = ppoints.point_inds[0]
        idx = trace.customdata[k]
        right_panel.value = _point_to_html(idx_to_point.get(idx), method)

    fig.data[0].on_hover(on_hover)

    # --- botao de alternancia de cor ---
    color_mode = widgets.ToggleButtons(
        options=["Classe", "Distorcao (SND)"],
        description="Cor:",
        style={"button_width": "auto"},
    )

    def on_color_change(change):
        if change["name"] != "value":
            return
        with fig.batch_update():
            if change["new"] == "Classe":
                fig.data[0].marker.color = class_colors
                fig.data[0].marker.colorscale = None
                fig.data[0].marker.showscale = False
                fig.layout.title = "Projecao (cor: classe)"
            else:
                fig.data[0].marker.color = snd_vals
                fig.data[0].marker.colorscale = "Spectral_r"
                fig.data[0].marker.cmin = cmin
                fig.data[0].marker.cmax = cmax
                fig.data[0].marker.cmid = cmid          # centro = SND 1
                fig.data[0].marker.showscale = True
                fig.data[0].marker.colorbar = dict(title="SND", thickness=12)
                fig.layout.title = "Projecao (cor: distorcao SND, centro=1)"

    color_mode.observe(on_color_change, names="value")

    # --- layout de duas telas ---
    left  = widgets.VBox([color_mode, fig],
                         layout=widgets.Layout(width="55%"))
    right = widgets.VBox([right_panel],
                         layout=widgets.Layout(width="45%"))
    ui = widgets.HBox([left, right],
                      layout=widgets.Layout(width="100%"))

    display(ui)
    return ui


# =============================================================================
# 5. HTML standalone (abre em aba/janela separada, tela cheia)
# =============================================================================

def _precompute_token_html_for_all(points, method):
    """
    Pre-renderiza o HTML dos heatmaps de todos os pontos (todas as features),
    para embutir no HTML standalone. Retorna dict {idx: html_do_painel}.

    Fazemos isso em Python (reusando o colormap do matplotlib) porque replicar
    os colormaps divergentes em JS seria trabalhoso e propenso a divergencia.
    O JS so troca qual bloco HTML exibir no hover — o render ja vem pronto.
    """
    return {p["idx"]: _point_to_html(p, method) for p in points if p is not None}


def build_standalone_html(meta, points, label_names, label_colors=None,
                          point_size=6, title="FADEx SAE — Visualizacao"):
    """
    Gera uma pagina HTML autossuficiente (Plotly via CDN + dados embutidos) com
    a interface de duas telas. O hover e tratado 100% em JavaScript, sem depender
    do kernel Python — por isso funciona quando servida numa aba separada.

    Retorna a string HTML completa.
    """
    import json as _json

    valid_pts = [p for p in points if p is not None]
    xs   = [p["x"] for p in valid_pts]
    ys   = [p["y"] for p in valid_pts]
    labs = [p["label"] for p in valid_pts]
    snds = [p["snd"] for p in valid_pts]
    idxs = [p["idx"] for p in valid_pts]

    if label_colors is None:
        import matplotlib
        from matplotlib import cm
        cmap = cm.get_cmap("tab20")
        label_colors = [matplotlib.colors.to_hex(cmap(k % 20))
                        for k in range(len(label_names))]
    class_colors = [label_colors[l] for l in labs]

    _, cmin, cmax, cmid = snd_to_color_values(np.array(snds), center=1.0)

    # HTML dos paineis pre-renderizado por ponto
    panels = _precompute_token_html_for_all(valid_pts, meta["method"])

    # Serializa dados para o JS
    data_js = _json.dumps({
        "xs": xs, "ys": ys, "class_colors": class_colors,
        "snd": snds, "idxs": idxs,
        "cmin": cmin, "cmax": cmax, "cmid": cmid,
        "point_size": point_size,
        "panels": panels,
        "method": meta["method"],
    })

    html = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>__TITLE__</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    html, body { margin:0; padding:0; height:100%; font-family: sans-serif; }
    #container { display:flex; height:100vh; width:100vw; }
    #left  { width:55%; height:100%; display:flex; flex-direction:column;
             border-right:1px solid #ddd; }
    #right { width:45%; height:100%; overflow-y:auto; padding:12px;
             box-sizing:border-box; }
    #controls { padding:8px 12px; border-bottom:1px solid #eee; }
    #plot { flex:1; }
    button { padding:6px 12px; margin-right:6px; cursor:pointer;
             border:1px solid #ccc; border-radius:4px; background:#f7f7f7; }
    button.active { background:#264653; color:white; border-color:#264653; }
    .hint { color:#999; font-style:italic; }
  </style>
</head>
<body>
  <div id="container">
    <div id="left">
      <div id="controls">
        <b>Cor:</b>
        <button id="btn-class" class="active" onclick="setMode('class')">Classe</button>
        <button id="btn-snd" onclick="setMode('snd')">Distorcao (SND)</button>
      </div>
      <div id="plot"></div>
    </div>
    <div id="right">
      <p class="hint">Passe o mouse sobre um ponto...</p>
    </div>
  </div>

  <script>
    const D = __DATA__;

    function makeTrace(mode) {
      const marker = { size: D.point_size, line: {width: 0} };
      if (mode === 'class') {
        marker.color = D.class_colors;
      } else {
        marker.color = D.snd;
        marker.colorscale = 'Spectral';
        marker.reversescale = true;      // Spectral_r: frio<1, quente>1
        marker.cmin = D.cmin;
        marker.cmax = D.cmax;
        marker.cmid = D.cmid;            // centro do divergente = SND 1
        marker.showscale = true;
        marker.colorbar = { title: 'SND', thickness: 12 };
      }
      return {
        x: D.xs, y: D.ys, mode: 'markers', type: 'scattergl',
        marker: marker, customdata: D.idxs, hoverinfo: 'none'
      };
    }

    const layout = {
      margin: {l:10, r:10, t:20, b:10},
      xaxis: {showticklabels:false, zeroline:false, showgrid:false},
      yaxis: {showticklabels:false, zeroline:false, showgrid:false},
      plot_bgcolor: 'white', dragmode: 'pan'
    };
    const config = { responsive: true, scrollZoom: true, displaylogo: false };

    Plotly.newPlot('plot', [makeTrace('class')], layout, config);

    const plotDiv = document.getElementById('plot');
    const right = document.getElementById('right');

    plotDiv.on('plotly_hover', function(ev) {
      const pt = ev.points[0];
      const idx = pt.customdata;
      const html = D.panels[idx];
      if (html) right.innerHTML = html;
    });

    function setMode(mode) {
      Plotly.react('plot', [makeTrace(mode)], layout, config);
      document.getElementById('btn-class').classList.toggle('active', mode==='class');
      document.getElementById('btn-snd').classList.toggle('active', mode==='snd');
      // re-attach hover (react recria os handlers)
      document.getElementById('plot').on('plotly_hover', function(ev) {
        const idx = ev.points[0].customdata;
        const html = D.panels[idx];
        if (html) right.innerHTML = html;
      });
    }
  </script>
</body>
</html>"""

    html = (html.replace("__TITLE__", title)
                .replace("__DATA__", data_js))
    return html


def serve_html(html_str, port=8700, path="/", mode="iframe", height=800):
    """
    Sobe um servidor HTTP simples no kernel servindo o HTML autossuficiente, e
    o expoe no Colab. Como o HTML nao depende do kernel (hover em JS puro), ele
    funciona tanto embutido num iframe quanto numa janela separada.

    Parametros
    ----------
    mode : "iframe" (default, RECOMENDADO) embute a interface inline num iframe
           grande — contorna o bloqueio de third-party cookies que costuma deixar
           a janela separada em branco. Use o botao de fullscreen do Colab (canto
           do output) para tela cheia.
           "window" gera um link que abre em aba/janela separada. Mais suscetivel
           a bloqueio de cookies, dependendo do navegador.
    height : altura do iframe em pixels (so no modo iframe).

    Uso:
        html = build_standalone_html(meta, points, label_names, label_colors)
        serve_html(html, mode="iframe", height=850)   # inline, grande
        # ou
        serve_html(html, mode="window")               # tenta abrir nova aba
    """
    import os
    import threading
    import http.server
    import socketserver

    serve_dir = "/content/_viz_serve"
    os.makedirs(serve_dir, exist_ok=True)
    with open(os.path.join(serve_dir, "index.html"), "w") as f:
        f.write(html_str)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=serve_dir, **kw)
        def log_message(self, *a):
            pass  # silencia logs

    # Servidor em thread daemon (nao bloqueia o notebook).
    # allow_reuse_address evita "Address already in use" ao re-rodar a celula.
    class ReusableServer(socketserver.TCPServer):
        allow_reuse_address = True

    def _serve():
        try:
            with ReusableServer(("", port), Handler) as httpd:
                httpd.serve_forever()
        except OSError:
            pass  # porta ja em uso (servidor de execucao anterior segue vivo)

    t = threading.Thread(target=_serve, daemon=True)
    t.start()

    from google.colab import output
    if mode == "iframe":
        output.serve_kernel_port_as_iframe(port, path=path, height=str(height))
    else:
        output.serve_kernel_port_as_window(
            port, path=path,
            anchor_text="🔗 Abrir visualizacao em nova janela")


# Alias retrocompatível
def serve_html_new_window(html_str, port=8700, path="/"):
    """Compat: chama serve_html no modo window."""
    return serve_html(html_str, port=port, path=path, mode="window")
