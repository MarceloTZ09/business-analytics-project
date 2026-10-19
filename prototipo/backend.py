"""Servidor local: formulario HTML y predicciones del Random Forest del equipo."""
import argparse
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import joblib
import pandas as pd
import sklearn

CARPETA = Path(__file__).resolve().parent
FEATURES = ["Category", "Target", "Duration", "Video_Header", "Videos_Bool",
            "Images_Bool", "Description", "Rewards", "Prev_Success", "Backed", "Links"]


def cargar_paquete():
    # Solo cargar el modelo confiable del equipo. No aceptar archivos del navegador.
    paquete = joblib.load(CARPETA / "random_forest_crowdfunding.joblib")
    if paquete.get("version_sklearn") != sklearn.__version__:
        raise RuntimeError("Instala las versiones indicadas en requirements.txt.")
    if paquete["features"] != FEATURES:
        raise RuntimeError("El modelo no tiene los predictores esperados.")
    return paquete


def validar(datos, categorias):
    if not isinstance(datos, dict) or set(datos) != set(FEATURES):
        raise ValueError("Completa exactamente los 11 campos del formulario.")
    if not isinstance(datos["Category"], str) or datos["Category"] not in categorias:
        raise ValueError("Selecciona una categoría válida.")
    limpios = {"Category": datos["Category"]}
    for nombre in FEATURES[1:]:
        valor = datos[nombre]
        if isinstance(valor, bool) or not isinstance(valor, (int, float)):
            raise ValueError(f"{nombre} debe ser un número.")
        if not math.isfinite(valor) or valor < 0:
            raise ValueError(f"{nombre} debe ser un número finito no negativo.")
        if nombre != "Target" and valor != int(valor):
            raise ValueError(f"{nombre} debe ser un entero.")
        if nombre in ("Video_Header", "Videos_Bool", "Images_Bool") and valor not in (0, 1):
            raise ValueError(f"{nombre} debe ser 0 o 1.")
        limpios[nombre] = float(valor) if nombre == "Target" else int(valor)
    if limpios["Target"] < 1 or limpios["Rewards"] < 1:
        raise ValueError("La meta y las recompensas deben ser al menos 1.")
    if not 2 <= limpios["Duration"] <= 60:
        raise ValueError("La duración debe estar entre 2 y 60 días.")
    return limpios


def crear_handler(paquete):
    class Handler(BaseHTTPRequestHandler):
        def responder(self, estado, datos):
            contenido = json.dumps(datos, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(estado)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(contenido)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(contenido)

        def do_GET(self):
            if self.path == "/api/config":
                self.responder(200, {"categorias": paquete["categorias"],
                                      "umbral": float(paquete.get("umbral", 0.5))})
            elif self.path in ("/", "/index.html"):
                contenido = (CARPETA / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(contenido)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(contenido)
            else:
                self.responder(404, {"error": "Página no encontrada."})

        def do_POST(self):
            if self.path != "/api/predecir":
                self.responder(404, {"error": "Ruta no encontrada."})
                return
            try:
                longitud = int(self.headers.get("Content-Length", "0"))
                if not 0 < longitud <= 16384:
                    raise ValueError("Solicitud vacía o demasiado grande.")
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("El formulario debe enviar JSON.")
                datos = json.loads(self.rfile.read(longitud).decode("utf-8"))
                entrada = validar(datos, paquete["categorias"])
            except (ValueError, UnicodeError):
                self.responder(400, {"error": "Datos inválidos: revisa los campos y sus límites."})
                return
            try:
                modelo = paquete["modelo"]
                tabla = pd.DataFrame([entrada])[paquete["features"]]
                indice = list(modelo.classes_).index(1)
                probabilidad = float(modelo.predict_proba(tabla)[0, indice])
                umbral = float(paquete.get("umbral", 0.5))
                self.responder(200, {"probabilidad_exito": probabilidad,
                    "clasificacion": "Exitosa" if probabilidad >= umbral else "No exitosa",
                    "umbral": umbral})
            except Exception:
                self.responder(500, {"error": "No se pudo calcular la predicción."})

    return Handler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    paquete = cargar_paquete()
    servidor = ThreadingHTTPServer(("127.0.0.1", args.port), crear_handler(paquete))
    print(f"Abre http://127.0.0.1:{args.port} en tu navegador. Ctrl+C para detener.", flush=True)
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        servidor.server_close()
