import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pandas as pd

from backend import cargar_paquete, crear_handler


class PruebasBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paquete = cargar_paquete()
        cls.servidor = ThreadingHTTPServer(("127.0.0.1", 0), crear_handler(cls.paquete))
        cls.hilo = threading.Thread(target=cls.servidor.serve_forever, daemon=True)
        cls.hilo.start()
        cls.url = f"http://127.0.0.1:{cls.servidor.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.servidor.shutdown()
        cls.servidor.server_close()
        cls.hilo.join()

    def post(self, datos):
        solicitud = urllib.request.Request(self.url + "/api/predecir",
            data=json.dumps(datos).encode(), headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(solicitud, timeout=10)

    def test_pagina_y_configuracion(self):
        with urllib.request.urlopen(self.url) as respuesta:
            self.assertIn(b"Evaluar", respuesta.read())
        with urllib.request.urlopen(self.url + "/api/config") as respuesta:
            self.assertEqual(json.load(respuesta)["categorias"], self.paquete["categorias"])

    def test_prediccion_coincide_con_pipeline(self):
        datos = {"Category": "Design", "Target": 10034.0, "Duration": 60,
            "Video_Header": 1, "Videos_Bool": 0, "Images_Bool": 1,
            "Description": 21, "Rewards": 4, "Prev_Success": 1, "Backed": 3, "Links": 1}
        with self.post(datos) as respuesta:
            salida = json.load(respuesta)
        modelo = self.paquete["modelo"]
        indice = list(modelo.classes_).index(1)
        directa = modelo.predict_proba(pd.DataFrame([datos])[self.paquete["features"]])[0, indice]
        self.assertAlmostEqual(salida["probabilidad_exito"], float(directa), places=12)
        self.assertEqual(salida["clasificacion"], "Exitosa" if directa >= .5 else "No exitosa")

    def test_entrada_invalida(self):
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.post({"Category": "inventada"})
        self.assertEqual(error.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
