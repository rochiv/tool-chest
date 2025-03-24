import os
import sys
import timeit
import zipfile
import logging
import argparse
from pathlib import Path


def setup_logging(log_level=logging.INFO, log_file=None):
    """Configure logging with the specified level and optional file output."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    handlers = []
    # Always add console handler
    handlers.append(logging.StreamHandler())

    # Add file handler if specified
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )

    return logging.getLogger(__name__)


def unzip_files(folder_path, pattern=".zip", output_dir=None, remove=False):
    """
    Unzip files in the specified folder matching the given pattern.

    Args:
        folder_path (str): Path to folder containing zip files
        pattern (str): File extension or pattern to match zip files
        output_dir (str): Directory to extract files to (default is same as zip file)
        remove (bool): Whether to remove zip files after extraction

    Returns:
        tuple: (success_count, error_count)
    """
    logger = logging.getLogger(__name__)

    # Convert to Path object for better path handling
    folder = Path(folder_path)

    # Verify folder exists
    if not folder.is_dir():
        logger.error(f"Folder not found: {folder}")
        return 0, 1

    logger.info(f"Processing zip files in: {folder}")

    # Find all zip files matching pattern
    zip_files = list(folder.glob(f"*{pattern}"))
    if not zip_files:
        logger.warning(
            f"No files matching pattern '*{pattern}' found in {folder}")
        return 0, 0

    logger.info(f"Found {len(zip_files)} files to process")

    # Process each file
    success_count = 0
    error_count = 0

    for zip_file in zip_files:
        try:
            # Determine extraction directory
            extract_dir = Path(output_dir) if output_dir else zip_file.parent

            logger.debug(f"Extracting {zip_file} to {extract_dir}")

            # Extract the file
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # Log file contents at debug level
                file_list = zip_ref.namelist()
                logger.debug(f"Archive contains {len(file_list)} files")

                # Extract all files
                zip_ref.extractall(path=extract_dir)

            logger.info(f"Successfully extracted: {zip_file}")
            success_count += 1

            # Remove zip file if requested
            if remove:
                logger.debug(f"Removing zip file: {zip_file}")
                zip_file.unlink()
                logger.info(f"Removed: {zip_file}")

        except zipfile.BadZipFile:
            logger.error(f"Invalid zip file: {zip_file}")
            error_count += 1
        except PermissionError:
            logger.error(f"Permission denied accessing: {zip_file}")
            error_count += 1
        except Exception as e:
            logger.error(
                f"Error processing {zip_file}: {str(e)}", exc_info=True)
            error_count += 1

    # Log summary
    logger.info(
        f"Operation completed: {success_count} files processed successfully, {error_count} errors")
    return success_count, error_count


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract zip files and optionally delete them")
    parser.add_argument(
        "folder_path", help="Path to folder containing zip files")
    parser.add_argument(
        "--pattern",
        default=".zip",
        help="File extension or pattern to match (default: .zip)"
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to extract files to (default is same as zip file)"
    )
    parser.add_argument(
        "--remove",
        action="store_true",
        help="Remove zip files after extraction"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level"
    )
    parser.add_argument(
        "--log-file",
        help="Save logs to specified file"
    )

    return parser.parse_args()


def main():
    """Main program entry point."""
    try:
        # Parse command line arguments
        args = parse_args()

        # Setup logging
        log_level = getattr(logging, args.log_level)
        logger = setup_logging(log_level=log_level, log_file=args.log_file)

        logger.debug("Starting unzip utility")
        logger.debug(f"Arguments: {args}")

        # Record start time for performance tracking
        start_time = timeit.default_timer()

        # Process files
        success_count, error_count = unzip_files(
            args.folder_path,
            pattern=args.pattern,
            output_dir=args.output_dir,
            remove=args.remove
        )

        # Calculate and log execution time
        elapsed = timeit.default_timer() - start_time
        logger.info(f"Time taken: {elapsed:.2f} seconds")

        # Return appropriate exit code
        return 1 if error_count > 0 else 0

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.critical(f"Unhandled exception: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    # Configure basic logging for startup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run main function and exit with appropriate status
    sys.exit(main())
