package ai.onnxruntime.example.objectdetection.utils
import android.graphics.*
import kotlin.math.max
import kotlin.math.min

object ImageProcessingUtils {
    /**
     * Generates a heatmap color based on value
     */
    fun getHeatMapColor(value: Float): Int {
        val normalizedValue = value.coerceIn(0f, 1f)
        return when {
            normalizedValue < 0.25f -> {
                // Blue to Cyan
                val blue = 255
                val green = (normalizedValue * 4 * 255).toInt()
                Color.rgb(0, green, blue)
            }
            normalizedValue < 0.5f -> {
                // Cyan to Green
                val factor = (normalizedValue - 0.25f) * 4
                val blue = ((1 - factor) * 255).toInt()
                Color.rgb(0, 255, blue)
            }
            normalizedValue < 0.75f -> {
                // Green to Yellow
                val factor = (normalizedValue - 0.5f) * 4
                val red = (factor * 255).toInt()
                Color.rgb(red, 255, 0)
            }
            else -> {
                // Yellow to Red
                val factor = (normalizedValue - 0.75f) * 4
                val green = ((1 - factor) * 255).toInt()
                Color.rgb(255, green, 0)
            }
        }
    }

    /**
     * Overlays heatmap on original image
     */
    fun overlayHeatmap(
        originalBitmap: Bitmap,
        heatmapValues: FloatArray,
        heatmapWidth: Int,
        heatmapHeight: Int,
        alpha: Int = 128
    ): Bitmap {
        val outputBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(outputBitmap)

        val scaleX = originalBitmap.width.toFloat() / heatmapWidth
        val scaleY = originalBitmap.height.toFloat() / heatmapHeight

        val paint = Paint().apply {
            style = Paint.Style.FILL
        }

        for (y in 0 until heatmapHeight) {
            for (x in 0 until heatmapWidth) {
                val index = y * heatmapWidth + x
                val value = heatmapValues[index]

                paint.color = getHeatMapColor(value)
                paint.alpha = (value * alpha).toInt().coerceIn(0, alpha)

                val rect = RectF(
                    x * scaleX,
                    y * scaleY,
                    (x + 1) * scaleX,
                    (y + 1) * scaleY
                )
                canvas.drawRect(rect, paint)
            }
        }

        return outputBitmap
    }

    /**
     * Draws bounding boxes with labels on image
     */
    fun drawDetections(
        bitmap: Bitmap,
        detections: List<Detection>,
        textSize: Float = 48f,
        strokeWidth: Float = 4f
    ): Bitmap {
        val outputBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(outputBitmap)

        val boxPaint = Paint().apply {
            color = Color.RED
            style = Paint.Style.STROKE
            this.strokeWidth = strokeWidth
        }

        val textPaint = Paint().apply {
            color = Color.WHITE
            this.textSize = textSize
            typeface = Typeface.DEFAULT_BOLD
        }

        val textBackgroundPaint = Paint().apply {
            color = Color.RED
            alpha = 128
            style = Paint.Style.FILL
        }

        for (detection in detections) {
            // Draw bounding box
            canvas.drawRect(detection.boundingBox, boxPaint)

            // Draw label background
            val textBounds = Rect()
            textPaint.getTextBounds(detection.label, 0, detection.label.length, textBounds)
            canvas.drawRect(
                detection.boundingBox.left,
                detection.boundingBox.top - textBounds.height() - 8,
                detection.boundingBox.left + textBounds.width() + 8,
                detection.boundingBox.top,
                textBackgroundPaint
            )

            // Draw label text
            canvas.drawText(
                detection.label,
                detection.boundingBox.left + 4,
                detection.boundingBox.top - 4,
                textPaint
            )
        }

        return outputBitmap
    }

    /**
     * Normalizes a float array to [0,1] range
     */
    fun normalizeArray(array: FloatArray): FloatArray {
        val min = array.minOrNull() ?: 0f
        val max = array.maxOrNull() ?: 1f
        val range = max - min

        return if (range != 0f) {
            array.map { (it - min) / range }.toFloatArray()
        } else {
            array.map { 0.5f }.toFloatArray()
        }
    }

    /**
     * Calculates Intersection over Union for two rectangles
     */
    private fun calculateIoU(a: RectF, b: RectF): Float {
        val intersection = RectF().apply {
            left = max(a.left, b.left)
            top = max(a.top, b.top)
            right = min(a.right, b.right)
            bottom = min(a.bottom, b.bottom)
        }

        if (intersection.left >= intersection.right ||
            intersection.top >= intersection.bottom) return 0f

        val intersectionArea = intersection.width() * intersection.height()
        val aArea = a.width() * a.height()
        val bArea = b.width() * b.height()
        val unionArea = aArea + bArea - intersectionArea

        return intersectionArea / unionArea
    }

    data class Detection(
        val label: String,
        val confidence: Float,
        val boundingBox: RectF
    )
}