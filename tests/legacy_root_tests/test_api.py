import time
import requests

url = "http://localhost:8000"
print("Esperando Uvicorn subir...")
connected = False
for _ in range(40):
    try:
        if requests.get(f"{url}/healtcheck").status_code == 200:
            connected = True
            break
    except:
        time.sleep(1)

if not connected:
     print("A API nunca subiu. Cancelando testes.")
     exit()

print("\\n--- TESTE 1: USUARIO CONHECIDO ---")
try:
    res1 = requests.get(f"{url}/recomendacoes/1?k=5")
    if res1.status_code == 200:
        print(f"Status OK: Encontrados {len(res1.json())} jogos.")
        for j in res1.json():
            print(f"- [ID {j['id']}] Score: {j['score']:.4f} | {j.get('titulo', 'Unknown')} | {j.get('genres', [])}")
    else:
        print("Falhou", res1.status_code)
except Exception as e:
    print("Erro API: ", e)

print("\\n--- TESTE 2: VARIACOES DOS MODOS ---")
try:
    for modo in ['aventureiro', 'equilibrado', 'conservador']:
        res2 = requests.get(f"{url}/recomendacoes/1?k=5&modo={modo}")
        if res2.status_code == 200:
            print(f"{modo.upper()}: {[j['id'] for j in res2.json()]}")
        else:
             print(f"{modo} error: ", res2.status_code)
except Exception as e:
    pass

print("\\n--- TESTE 4: COBERTURA MANUAL ---")
try:
    ids_unicos = set()
    errors = 0
    # Amostra de 100 usuarios (id 10 ao 110)
    for u in range(10, 110):
        re = requests.get(f"{url}/recomendacoes/{u}?k=5")
        if re.status_code == 200:
            ids_unicos.update([j['id'] for j in re.json()])
        else:
            errors += 1
    print(f"Total recomendados unicos (100 users / k=5): {len(ids_unicos)}")
    # Como não temos catálogo exato de id salvo na DB dummy de backend o numero % exato eh base pro /122507
    print(f"Erros de timeout: {errors}")
except Exception as e:
    pass

print("\\n--- TESTE 5: COLD START ---")
try:
    r_cs = requests.get(f"{url}/recomendacoes/99999?k=5")
    if r_cs.status_code == 200:
        print("SUCESSO (Fallback Funciona). Jogos Frios:")
        print([j['id'] for j in r_cs.json()])
    else:
        print("Falha COLD START", r_cs.status_code)
except Exception as e:
    pass

print("\\n==== Teste API Local Realizado ====")
