import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from communication.pi_client import PiClient

def main():
    text = input("Nhập lệnh cho robot: ")
    client = PiClient()
    client.send_command(text)

if __name__ == "__main__":
    main()
