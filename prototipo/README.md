# Formulario HTML + backend Python

Frontend: index.html (HTML/CSS/JavaScript).
Backend: backend.py (servidor estándar de Python y el modelo real del equipo).
No utiliza Streamlit, Flask, una API externa ni datos simulados.

## Probar con el entorno que ya instalaste

1. Descomprime esta carpeta. Conserva juntos los cuatro archivos de código/modelo.
2. Abre CMD en esta carpeta.
3. Si el entorno anterior sigue en la misma ubicación, ejecuta:

```cmd
"C:\Users\jairo\OneDrive\Escritorio\prototipo_crowdfunding\.venv\Scripts\python.exe" backend.py
```

4. Abre http://127.0.0.1:8000 en el navegador. No abras el HTML con doble clic.
5. Conserva CMD abierto; Ctrl+C detiene el servidor.

## Otra computadora

Requiere Python 3.11 o 3.12 y las dependencias de requirements.txt.
No copia ni incluye el entorno .venv de la aplicación anterior.

## Límites

Servidor de demostración local: escucha únicamente en 127.0.0.1; no está
preparado para desplegarse directamente como un servicio público.
La predicción usa el joblib del equipo, sin reentrenar ni cambiar el umbral.
No cargues archivos joblib de origen desconocido.
Los montos y Description requieren las unidades/definiciones originales.
Sigue pendiente verificar las fechas de medición de los predictores al lanzamiento.
Las probabilidades no tienen una evaluación de calibración documentada.
