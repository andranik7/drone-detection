"""Unit tests for data preprocessing."""


from src.data.preprocess import validate_label, validate_image


def test_validate_label_valid(tmp_path):
    label_file = tmp_path / "test.txt"
    label_file.write_text("0 0.5 0.5 0.3 0.3\n")
    assert validate_label(label_file) is True


def test_validate_label_invalid_format(tmp_path):
    label_file = tmp_path / "test.txt"
    label_file.write_text("0 0.5 0.5\n")  # missing fields
    assert validate_label(label_file) is False


def test_validate_label_out_of_range(tmp_path):
    label_file = tmp_path / "test.txt"
    label_file.write_text("0 1.5 0.5 0.3 0.3\n")  # >1.0
    assert validate_label(label_file) is False


def test_validate_image_invalid(tmp_path):
    bad_img = tmp_path / "bad.jpg"
    bad_img.write_text("not an image")
    assert validate_image(bad_img) is False
