
import json
import time
import requests

url = "http://127.0.0.1:9931/completion"

payload = {
    "prompt": "Explain what a Large Language Model is in simple terms.",
    "n_predict": 16,
    "temperature": 0,
    "stream": True
}

start = time.perf_counter()

with requests.post(
    url,
    json=payload,
    stream=True,
    timeout=(10, 180)
) as response:

    response.raise_for_status()

    print("HTTP response received.")

    for line in response.iter_lines(decode_unicode=True):

        elapsed = time.perf_counter() - start

        if not line:
            continue

        print(f"\n[{elapsed:.6f}s] RAW EVENT:")
        print(line)

        if line.startswith("data:"):

            data = line[len("data:"):].strip()

            if data == "[DONE]":
                print("\nStream finished.")
                break

            try:
                event = json.loads(data)

                print("Parsed content:", repr(event.get("content")))
                print("Stop:", event.get("stop"))

                if "timings" in event:
                    print("Timings:", event["timings"])

            except json.JSONDecodeError:
                print("Could not parse event as JSON.")
