import argparse
import qrcode
from PIL import Image


def generate_qrcode(url_path: str):
    # Create a QR code instance
    qr = qrcode.QRCode(
        version=1,  # QR code version (adjust as needed)
        error_correction=qrcode.constants.ERROR_CORRECT_L,  # Error correction level
        box_size=10,  # Size of each QR code "box"
        border=4,  # Border width (adjust as needed)
    )

    # Add data to the QR code
    qr.add_data(url_path)
    qr.make(fit=True)

    # Create an image from the QR code instance
    qr_img = qr.make_image(fill_color="black", back_color="white")

    return qr_img


def save_qrcode(qr_image, qr_code_name: str) -> None:
    print("QR code saved!")
    qr_image.save(f"{qr_code_name}.png")


def main():
    parser = argparse.ArgumentParser(description='Generate and display a QR code for a given URL.')
    parser.add_argument('url', type=str, help='URL to encode in the QR code')
    parser.add_argument('--name', '-n', type=str, default='qr_code',
                        help='Name of the output QR code image (default: qr_code)')

    args = parser.parse_args()

    url_path = args.url
    img = generate_qrcode(url_path)
    save_qrcode(img, args.name)


if __name__ == "__main__":
    main()
