package ai.onnxruntime.example.objectdetection.processing

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.example.objectdetection.models.DetectionModel
import ai.onnxruntime.example.objectdetection.utils.BitmapUtils
import ai.onnxruntime.example.objectdetection.utils.ImageProcessingUtils
import android.graphics.Bitmap
import java.nio.FloatBuffer

import android.util.Log

class PatchcoreProcessing(
    private val config: DetectionModel.Patchcore = DetectionModel.Patchcore()
) : ProcessingStrategy {

    data class PatchcoreResult(
        val originalBitmap: Bitmap,
        val rawScore: Float,
        val predLabel: String,
        val anomalyMap: FloatArray,
        val pixelThreshold: Float,
        val anomalousPixelPercentage: Float
    )

    companion object {
        private const val IMAGE_THRESHOLD = 46f  // From original code
        private const val ANOMALY_PIXEL_PERCENTAGE_THRESHOLD = 0.3f
        private const val MIN_SCORE = 0f
        private const val MAX_SCORE = 1f
    }

    override fun preprocess(bitmap: Bitmap): OnnxTensor {
        try {
            // Use BitmapUtils for preprocessing
            val processedBitmap = BitmapUtils.preprocess(
                bitmap = bitmap,
                targetWidth = config.inputSize.first,
                targetHeight = config.inputSize.second,
                shouldCropToSquare = false  // Adjust based on your needs
            )

            // Convert to CHW format with normalization
            val floatArray = BitmapUtils.bitmapToFloatArray(
                bitmap = processedBitmap,
                mean = floatArrayOf(0f, 0f, 0f),  // Adjust normalization values if needed
                std = floatArrayOf(255f, 255f, 255f)
            )

            Log.d("PatchcoreProcessing", "Preprocessed tensor shape: [1, 3, ${config.inputSize.first}, ${config.inputSize.second}]")

            return OnnxTensor.createTensor(
                OrtEnvironment.getEnvironment(),
                FloatBuffer.wrap(floatArray),
                longArrayOf(1, 3, config.inputSize.first.toLong(), config.inputSize.second.toLong())
            ).also {
                // Clean up the temporary bitmap if it's different from the input
                if (processedBitmap != bitmap) {
                    processedBitmap.recycle()
                }
            }
        } catch (e: Exception) {
            Log.e("PatchcoreProcessing", "Error in preprocessing: ${e.message}")
            throw e
        }
    }

    override fun postprocess(outputs: Map<String, OnnxTensor>): DetectionResult {
        try {
            // Extract anomaly map and score from outputs
            val anomalyMapTensor = outputs["anomaly_map"] as? OnnxTensor
                ?: throw IllegalStateException("No anomaly map in output")
            val scoreTensor = outputs["score"] as? OnnxTensor
                ?: throw IllegalStateException("No score in output")

            // Get raw values
            val anomalyMap = anomalyMapTensor.floatBuffer.array()
            val rawScore = scoreTensor.floatBuffer.get(0)

            Log.d("PatchcoreProcessing", "Raw score: $rawScore")

            // Normalize anomaly map to [0, 1] range
            val normalizedMap = ImageProcessingUtils.normalizeArray(anomalyMap)

            // Calculate anomalous pixel percentage
            val anomalousPixelPercentage = normalizedMap.count {
                it > config.threshold
            }.toFloat() / normalizedMap.size

            // Determine if anomalous based on score
            val isAnomaly = rawScore > IMAGE_THRESHOLD
            val predLabel = if (isAnomaly) "Anomalous" else "Normal"

            return DetectionResult(
                score = normalizeScore(rawScore),
                label = predLabel,
                heatmap = normalizedMap,
                threshold = config.threshold,
                confidence = anomalousPixelPercentage
            )
        } catch (e: Exception) {
            Log.e("PatchcoreProcessing", "Error in postprocessing: ${e.message}")
            throw e
        } finally {
            // Clean up tensors
            outputs.values.forEach { it.close() }
        }
    }

    private fun normalizeScore(value: Float): Float {
        return ((value - IMAGE_THRESHOLD) / (MAX_SCORE - MIN_SCORE) + 0.5f)
            .coerceIn(0f, 1f)
    }

    override fun getInputShape(): Array<Long> = arrayOf(
        1,  // batch size
        3,  // channels (RGB)
        config.inputSize.first.toLong(),  // height
        config.inputSize.second.toLong()  // width
    )

    override fun visualizeResult(
        originalBitmap: Bitmap,
        result: DetectionResult
    ): Bitmap {
        return ImageProcessingUtils.overlayHeatmap(
            originalBitmap = originalBitmap,
            heatmapValues = result.heatmap,
            heatmapWidth = config.inputSize.first,
            heatmapHeight = config.inputSize.second,
            alpha = 128
        )
    }
}

// Data class for detection results
data class DetectionResult(
    val score: Float,
    val label: String,
    val heatmap: FloatArray,
    val threshold: Float,
    val confidence: Float
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as DetectionResult

        if (score != other.score) return false
        if (label != other.label) return false
        if (!heatmap.contentEquals(other.heatmap)) return false
        if (threshold != other.threshold) return false
        if (confidence != other.confidence) return false

        return true
    }

    override fun hashCode(): Int {
        var result = score.hashCode()
        result = 31 * result + label.hashCode()
        result = 31 * result + heatmap.contentHashCode()
        result = 31 * result + threshold.hashCode()
        result = 31 * result + confidence.hashCode()
        return result
    }

}



