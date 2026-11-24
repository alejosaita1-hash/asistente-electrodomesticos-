"""Carga los manuales (data/paginas.jsonl) en Pinecone. Se corre UNA vez.
Uso:  PINECONE_API_KEY=tu_clave python cargar_pinecone.py
(En Windows PowerShell:  $env:PINECONE_API_KEY="tu_clave"; python cargar_pinecone.py)"""
import json, os, time
from pinecone import Pinecone

INDEX, NAMESPACE = "manuales-electro", "manuales"
CHUNK, SOLAPE = 1200, 200

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
if not pc.has_index(INDEX):
    pc.create_index_for_model(
        name=INDEX, cloud="aws", region="us-east-1",
        embed={"model": "llama-text-embed-v2", "field_map": {"text": "texto"}},
    )
idx = pc.Index(INDEX)

registros = []
with open(os.path.join("data", "paginas.jsonl"), encoding="utf-8") as f:
    for linea in f:
        p = json.loads(linea)
        t, i, n = p["texto"], 0, 0
        while i < len(t):
            trozo = t[i:i + CHUNK]
            if len(trozo.strip()) > 40:
                registros.append({
                    "_id": f"{p['producto_id']}-p{p['pagina']}-{n}",
                    "texto": trozo, "producto_id": p["producto_id"],
                    "marca": p["marca"], "modelo": p["modelo"],
                    "categoria": p["categoria"], "pagina": p["pagina"],
                })
                n += 1
            i += CHUNK - SOLAPE

print(f"{len(registros)} fragmentos a subir...")
for k in range(0, len(registros), 90):
    idx.upsert_records(NAMESPACE, registros[k:k + 90])
    print(f"  {min(k + 90, len(registros))}/{len(registros)}")
    time.sleep(1)
print("Listo. Esperá ~1 minuto y probá la app.")
