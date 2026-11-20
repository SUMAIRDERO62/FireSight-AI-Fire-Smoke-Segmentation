from firesight.evaluation.metrics import accuracy


if __name__ == "__main__":
    print(f"Baseline self-check accuracy: {accuracy(['normal'], ['normal']):.2f}")

