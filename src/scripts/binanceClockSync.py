import time
from binance.client import Client

# Initialize Binance client (no API keys needed for server time)
client = Client()

def get_binance_server_time():
    return client.get_server_time()["serverTime"]

def get_local_time_ms():
    return int(time.time() * 1000)

def main(iterations=100, delay=1):
    diffs = []

    print(f"Checking time difference between local PC and Binance server ({iterations} times)...\n")

    for i in range(iterations):
        local_time = get_local_time_ms()
        server_time = get_binance_server_time()

        diff = local_time - server_time  # positive means local clock ahead, negative means behind
        diffs.append(diff)

        print(f"Check {i+1}: Local time is {diff} ms {'ahead' if diff > 0 else 'behind'} Binance server time")

        time.sleep(delay)

    print("\nSummary:")
    print(f"Min difference: {min(diffs)} ms")
    print(f"Max difference: {max(diffs)} ms")
    print(f"Average difference: {sum(diffs)/len(diffs):.2f} ms")

if __name__ == "__main__":
    main()
