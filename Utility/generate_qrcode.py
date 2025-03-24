import logging
import argparse
import qrcode
from PIL import Image

# Create a module-specific logger
logger = logging.getLogger(__name__)


def generate_qrcode(url_path: str) -> Image.Image:
    # Use INFO for main application functions
    logger.info(f"Generating QR code for URL: {url_path}")

    try:
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

        # Use DEBUG for detailed implementation steps
        logger.debug(
            f"QR code created with version={qr.version}, box_size=10, border=4")

        # Create an image from the QR code instance
        qr_img = qr.make_image(fill_color="black", back_color="white")

        logger.info(f"Successfully generated QR code for {url_path}")
        return qr_img

    except Exception as e:
        # Use ERROR for operation failures
        logger.error(
            f"Failed to generate QR code for {url_path}: {str(e)}", exc_info=True)
        raise


def save_qrcode(qr_image: Image.Image, qr_code_name: str) -> None:
    file_path = f"{qr_code_name}.png"
    logger.info(f"Saving QR code to {file_path}")

    try:
        # save the image
        qr_image.save(file_path)
        logger.info(f"Successfully saved QR code to {file_path}")
    except Exception as e:
        # Use ERROR for file operation failures
        logger.error(
            f"Failed to save QR code to {file_path}: {str(e)}", exc_info=True)
        raise


def main():
    # Configure logging when run as a script
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    parser = argparse.ArgumentParser(
        description='Generate and display a QR code for a given URL.')
    parser.add_argument('url', type=str, help='URL to encode in the QR code')
    parser.add_argument('--name', '-n', type=str, default='qr_code',
                        help='Name of the output QR code image (default: qr_code)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')

    args = parser.parse_args()

    # Set debug level if requested
    if args.debug:
        logging.getLogger(__name__).setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    url_path = args.url
    logger.info(f"Starting QR code generation process for URL: {url_path}")

    try:
        img = generate_qrcode(url_path)
        save_qrcode(img, args.name)
        logger.info(f"QR code process completed successfully for {url_path}")
    except Exception as e:
        logger.critical(f"QR code generation process failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
