
import json
import time
import requests

def main():
    SERVER_URL = "http://127.0.0.1:9931/completion"

    payload = {
        "prompt": "Explain what a Large Language Model is in simple terms.",
        "n_predict": 64,
        "temperature": 0,
        "stream": True
    }

    print("Sending streaming request...")

    start_time = time.perf_counter()

    response = requests.post(
        SERVER_URL,
        json=payload,
        stream=True,
        timeout=(10, 180)
    )

    response.raise_for_status()

    first_token_time = None
    generated_content = ""
    final_data = None

    print("\n========== GENERATED RESPONSE ==========\n")

    for line in response.iter_lines(decode_unicode=True):

        if not line:
            continue

        if line.startswith("data:"):
            data_text = line[len("data:"):].strip()

            if data_text == "[DONE]":
                break

            try:
                event = json.loads(data_text)
            except json.JSONDecodeError:
                continue

            content = event.get("content", "")
            if content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()

                generated_content += content
                print(content, end="", flush=True)

            if event.get("stop") is True:
                final_data = event

    end_time = time.perf_counter()

    response.close()

    print("\n\n========== STREAMING MEASUREMENTS ==========")

    if first_token_time is not None:

        ttft = first_token_time - start_time

        print(f"First content received after: {ttft:.4f} seconds")

    else:
        print("No generated content was received.")

    print(f"Total request time: {end_time - start_time:.4f} seconds")

    print(f"Generated text length: {len(generated_content)} characters")

    if final_data and "timings" in final_data:

        timings = final_data["timings"]

        print("\nServer timing information:")

        print(f"Prompt processing: {timings.get('prompt_ms')} ms")

        print(f"Generation time: {timings.get('predicted_ms')} ms")

        print(f"Generated tokens: {timings.get('predicted_n')}")

        print(
            f"Server TPOT: "
            f"{timings.get('predicted_per_token_ms')} ms/token"
        )

    else:
        print("\nFinal server timing information was not found.")
        print("We will inspect the returned events before proceeding.")


if __name__ == "__main__":
    main()
