import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.command_normalizer import CommandNormalizer


def test_command_normalizer():
    normalizer = CommandNormalizer()

    test_cases = [
        # --- FORWARD (đi thẳng) ---
        ("Đi thẳng", "đi thẳng"),
        ("Đi thang", "đi thẳng"),
        ("Đi than", "đi thẳng"),
        ("đi thẳn", "đi thẳng"),
        ("tiến lên", "đi thẳng"),
        ("đi về phía trước", "đi thẳng"),
        ("robot tiến lên", "đi thẳng"),
        ("tiếp tục đi", "đi thẳng"),
        ("chạy về phía trước", "đi thẳng"),
        ("robot ơi đi thẳng giúp tôi", "đi thẳng"),
        ("bạn hãy đi về phía trước", "đi thẳng"),
        ("bây giờ đi thẳng nhé", "đi thẳng"),

        # --- BACKWARD (đi lùi) ---
        ("Đi lùi", "đi lùi"),
        ("lùi lại", "đi lùi"),
        ("đi về phía sau", "đi lùi"),
        ("hãy lùi lại một chút", "đi lùi"),
        ("tiến ngược lại", "đi lùi"),
        ("robot lùi", "đi lùi"),

        # --- LEFT (rẽ trái) ---
        ("rẽ trái", "rẽ trái"),
        ("rẽ trai", "rẽ trái"),
        ("quẹo trái", "rẽ trái"),
        ("rẽ chái", "rẽ trái"),
        ("quẹo trai", "rẽ trái"),
        ("quay sang trái", "rẽ trái"),
        ("làm ơn rẽ trái", "rẽ trái"),

        # --- RIGHT (rẽ phải) ---
        ("rẽ phải", "rẽ phải"),
        ("rẽ phai", "rẽ phải"),
        ("quẹo phải", "rẽ phải"),
        ("rẽ pai", "rẽ phải"),
        ("quẹo phai", "rẽ phải"),
        ("quay sang phải", "rẽ phải"),
        ("giúp tôi rẽ phải", "rẽ phải"),

        # --- STOP (dừng) ---
        ("dừng", "dừng"),
        ("dừng lại", "dừng"),
        ("đứng lại ngay", "dừng"),
        ("đi thẳng rồi dừng", "dừng"),
        ("ngừng lại", "dừng"),
        ("dừng robot", "dừng"),
        ("đừng đi nữa", "dừng"),
        ("không đi nữa", "dừng"),
        ("robot đứng lại ngay", "dừng"),

        # --- NON-COMMANDS (None) ---
        ("phía trước có người", None),
        ("bên trái có cửa", None),
        ("bên phải có người", None),
        ("xin chào robot", None),
        ("bạn khỏe không", None),
        ("tôi muốn nói chuyện", None),
        ("thời tiết hôm nay thế nào", None),
    ]

    passed = 0
    failed = 0

    print("\n" + "=" * 60)
    print("        BỘ KIỂM THỬ CHUẨN HÓA LỆNH TIẾNG VIỆT (NORMALIZER TEST)")
    print("=" * 60)

    for text, expected in test_cases:
        actual = normalizer.normalize(text)
        status = "PASS" if actual == expected else "FAIL"
        if status == "PASS":
            passed += 1
            print(f" [PASS] Input: '{text}' -> Output: {repr(actual)}")
        else:
            failed += 1
            print(f" [FAIL] Input: '{text}' -> Expected: {repr(expected)}, Got: {repr(actual)}")

    print("=" * 60)
    print(f" KẾT QUẢ TEST: {passed}/{len(test_cases)} PASS | {failed} FAIL")
    print("=" * 60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = test_command_normalizer()
    sys.exit(0 if success else 1)
