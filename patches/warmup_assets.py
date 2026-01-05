#!/usr/bin/env python3
"""
BabelDOC assets warmup script.

Downloads all fonts and models required by BabelDOC during Docker build,
so they are cached in the image and don't need to be downloaded at runtime.
"""

import asyncio
import logging
import os
import sys

# Set longer timeout for downloads
os.environ.setdefault("HTTPX_TIMEOUT", "600")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def warmup_with_retry(max_retries=3):
    """Run the warmup with retries."""
    import httpx
    from babeldoc.assets import assets

    for attempt in range(max_retries):
        try:
            logger.info(f"Warmup attempt {attempt + 1}/{max_retries}")

            # Create client with extended timeout
            timeout = httpx.Timeout(600.0, connect=60.0)

            async with httpx.AsyncClient(timeout=timeout) as client:
                # Download all fonts
                logger.info("Downloading all fonts...")
                await assets.download_all_fonts_async(client)

                # Download DocLayout ONNX model
                logger.info("Downloading DocLayout ONNX model...")
                await assets.get_doclayout_onnx_model_path_async(client)

                # Download table detection model
                logger.info("Downloading table detection model...")
                try:
                    await assets.get_table_detection_rapidocr_model_path_async(client)
                except Exception as e:
                    logger.warning(f"Table detection model download failed (non-critical): {e}")

            logger.info("All assets downloaded successfully!")
            return True

        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 30
                logger.info(f"Waiting {wait_time} seconds before retry...")
                await asyncio.sleep(wait_time)
            else:
                logger.error("All retry attempts exhausted")
                return False

    return False


def main():
    """Main entry point."""
    logger.info("Starting BabelDOC assets warmup...")

    try:
        success = asyncio.run(warmup_with_retry())
        if success:
            logger.info("Warmup completed successfully")
            return 0
        else:
            logger.error("Warmup failed after all retries")
            return 1
    except Exception as e:
        logger.error(f"Warmup failed with exception: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
