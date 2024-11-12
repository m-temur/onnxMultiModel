package ai.onnxruntime.example.objectdetection.utils
import android.graphics.*
import android.graphics.Matrix
import android.media.Image
import android.graphics.ImageFormat
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer

object BitmapUtils {

    /**
     * Converts ImageProxy to Bitmap
     */
    fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        val image = imageProxy.image ?: return null

        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(
            nv21,
            ImageFormat.NV21,
            image.width,
            image.height,
            null
        )

        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(
            Rect(0, 0, image.width, image.height),
            100,
            out
        )

        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    /**
     * Converts ImageProxy to Bitmap with size optimization
     */
    fun imageProxyToBitmapWithResize(imageProxy: ImageProxy, targetSize: Int): Bitmap? {
        val image = imageProxy.image ?: return null

        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(
            nv21,
            ImageFormat.NV21,
            image.width,
            image.height,
            null
        )

        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(
            Rect(0, 0, image.width, image.height),
            75, // Reduced quality for better performance
            out
        )

        val imageBytes = out.toByteArray()

        // Calculate target size while maintaining aspect ratio
        val (targetWidth, targetHeight) = calculateTargetSize(
            image.width,
            image.height,
            targetSize
        )

        // Decode with scaled dimensions
        val options = BitmapFactory.Options().apply {
            inJustDecodeBounds = true
        }
        BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size, options)

        options.apply {
            inJustDecodeBounds = false
            inSampleSize = calculateInSampleSize(options, targetWidth, targetHeight)
        }

        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size, options)
    }

    private fun calculateTargetSize(width: Int, height: Int, targetSize: Int): Pair<Int, Int> {
        val ratio = width.toFloat() / height.toFloat()
        return if (width > height) {
            Pair(targetSize, (targetSize / ratio).toInt())
        } else {
            Pair((targetSize * ratio).toInt(), targetSize)
        }
    }

    /**
     * Resizes a bitmap to the specified dimensions while maintaining aspect ratio
     */
    fun resizeBitmap(bitmap: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap {
        val scaleWidth = targetWidth.toFloat() / bitmap.width
        val scaleHeight = targetHeight.toFloat() / bitmap.height
        val scaleFactor = scaleWidth.coerceAtMost(scaleHeight)

        val matrix = Matrix().apply {
            postScale(scaleFactor, scaleFactor)
        }

        return Bitmap.createBitmap(
            bitmap,
            0,
            0,
            bitmap.width,
            bitmap.height,
            matrix,
            true
        )
    }

    /**
     * Rotates a bitmap by the specified degrees
     */
    fun rotateBitmap(bitmap: Bitmap, degrees: Int): Bitmap {
        if (degrees == 0) return bitmap

        val matrix = Matrix().apply {
            postRotate(degrees.toFloat())
        }

        return Bitmap.createBitmap(
            bitmap,
            0,
            0,
            bitmap.width,
            bitmap.height,
            matrix,
            true
        )
    }

    /**
     * Converts YUV_420_888 image to bitmap
     */
    fun imageToBitmap(image: Image): Bitmap? {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(
            nv21,
            ImageFormat.NV21,
            image.width,
            image.height,
            null
        )

        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(
            Rect(0, 0, image.width, image.height),
            100,
            out
        )

        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    /**
     * Calculates the optimal sample size for loading a bitmap
     */
    fun calculateInSampleSize(
        options: BitmapFactory.Options,
        reqWidth: Int,
        reqHeight: Int
    ): Int {
        val (height: Int, width: Int) = options.run { outHeight to outWidth }
        var inSampleSize = 1

        if (height > reqHeight || width > reqWidth) {
            val halfHeight: Int = height / 2
            val halfWidth: Int = width / 2

            while (halfHeight / inSampleSize >= reqHeight &&
                halfWidth / inSampleSize >= reqWidth) {
                inSampleSize *= 2
            }
        }
        return inSampleSize
    }

    /**
     * Crops bitmap to center square
     */
    fun cropToSquare(bitmap: Bitmap): Bitmap {
        val dimension = bitmap.width.coerceAtMost(bitmap.height)
        val x = (bitmap.width - dimension) / 2
        val y = (bitmap.height - dimension) / 2

        return Bitmap.createBitmap(
            bitmap,
            x,
            y,
            dimension,
            dimension
        )
    }

    /**
     * Converts bitmap to ByteBuffer for model input
     */
    fun bitmapToByteBuffer(
        bitmap: Bitmap,
        width: Int,
        height: Int,
        mean: Float = 0f,
        std: Float = 255f
    ): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(4 * width * height * 3)
        byteBuffer.order(java.nio.ByteOrder.nativeOrder())

        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        for (pixel in pixels) {
            val r = (pixel shr 16 and 0xFF)
            val g = (pixel shr 8 and 0xFF)
            val b = (pixel and 0xFF)

            // Normalize pixel values
            byteBuffer.putFloat(((r - mean) / std))
            byteBuffer.putFloat(((g - mean) / std))
            byteBuffer.putFloat(((b - mean) / std))
        }

        return byteBuffer
    }

    /**
     * Converts bitmap to FloatArray in CHW format
     */
    fun bitmapToFloatArray(
        bitmap: Bitmap,
        mean: FloatArray = floatArrayOf(0f, 0f, 0f),
        std: FloatArray = floatArrayOf(255f, 255f, 255f)
    ): FloatArray {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        val floatValues = FloatArray(width * height * 3)
        var index = 0

        // Convert to CHW format
        for (channel in 0..2) {
            for (y in 0 until height) {
                for (x in 0 until width) {
                    val pixel = pixels[y * width + x]
                    val value = when (channel) {
                        0 -> pixel shr 16 and 0xFF // R
                        1 -> pixel shr 8 and 0xFF  // G
                        2 -> pixel and 0xFF        // B
                        else -> 0
                    }
                    floatValues[index++] = ((value - mean[channel]) / std[channel])
                }
            }
        }

        return floatValues
    }

    /**
     * Applies preprocessing transformations to bitmap
     */
    fun preprocess(
        bitmap: Bitmap,
        targetWidth: Int,
        targetHeight: Int,
        rotation: Int = 0,
        shouldCropToSquare: Boolean = false
    ): Bitmap {
        var processedBitmap = bitmap

        if (shouldCropToSquare) {
            processedBitmap = cropToSquare(processedBitmap)
        }

        if (rotation != 0) {
            processedBitmap = rotateBitmap(processedBitmap, rotation)
        }

        if (processedBitmap.width != targetWidth || processedBitmap.height != targetHeight) {
            processedBitmap = resizeBitmap(processedBitmap, targetWidth, targetHeight)
        }

        return processedBitmap
    }
}