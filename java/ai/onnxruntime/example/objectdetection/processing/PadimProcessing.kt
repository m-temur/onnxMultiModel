package ai.onnxruntime.example.objectdetection.processing

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.example.objectdetection.models.DetectionModel
import android.graphics.Bitmap
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.example.objectdetection.models.DetectionResult.Companion.LABEL_ANOMALOUS
import ai.onnxruntime.example.objectdetection.models.DetectionResult.Companion.LABEL_NORMAL
import ai.onnxruntime.example.objectdetection.utils.BitmapUtils
import ai.onnxruntime.example.objectdetection.utils.ImageProcessingUtils
import java.nio.FloatBuffer
import android.util.Log

class PadimProcessing(
    private val config: DetectionModel.Padim
) : ProcessingStrategy {

    override fun preprocess(bitmap: Bitmap): OnnxTensor {
        try {
            // Use BitmapUtils for preprocessing
            val processedBitmap = BitmapUtils.preprocess(
                bitmap = bitmap,
                targetWidth = config.inputSize.first,
                targetHeight = config.inputSize.second,
                shouldCropToSquare = false
            )

            // Convert to CHW format with normalization
            val floatArray = BitmapUtils.bitmapToFloatArray(
                bitmap = processedBitmap,
                mean = floatArrayOf(0.485f * 255, 0.456f * 255, 0.406f * 255),  // ImageNet normalization
                std = floatArrayOf(0.229f * 255, 0.224f * 255, 0.225f * 255)    // ImageNet normalization
            )

            Log.d("PadimProcessing", "Preprocessed tensor shape: [1, 3, ${config.inputSize.first}, ${config.inputSize.second}]")

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
            Log.e("PadimProcessing", "Error in preprocessing: ${e.message}")
            throw e
        }
    }

    override fun postprocess(outputs: Map<String, OnnxTensor>): DetectionResult {
        try {
            // Extract anomaly map and score from outputs
            val anomalyMapTensor = outputs["anomaly_map"] as? OnnxTensor
                ?: throw IllegalStateException("No anomaly map in output")
            val scoreTensor = outputs["anomaly_score"] as? OnnxTensor
                ?: throw IllegalStateException("No score in output")

            // Get raw values
            val anomalyMap = anomalyMapTensor.floatBuffer.array()
            val rawScore = scoreTensor.floatBuffer.get(0)

            Log.d("PadimProcessing", "Raw score: $rawScore")

            // Normalize anomaly map to [0, 1] range
            val normalizedMap = ImageProcessingUtils.normalizeArray(anomalyMap)

            // Calculate anomalous pixel percentage
            val anomalousPixelPercentage = normalizedMap.count {
                it > config.threshold
            }.toFloat() / normalizedMap.size

            // Determine if anomalous based on score
            val isAnomaly = rawScore > config.threshold
            val predLabel = if (isAnomaly) LABEL_ANOMALOUS else LABEL_NORMAL

            return DetectionResult(
                score = rawScore,
                label = predLabel,
                heatmap = normalizedMap,
                threshold = config.threshold,
                confidence = anomalousPixelPercentage
            )
        } catch (e: Exception) {
            Log.e("PadimProcessing", "Error in postprocessing: ${e.message}")
            throw e
        }
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