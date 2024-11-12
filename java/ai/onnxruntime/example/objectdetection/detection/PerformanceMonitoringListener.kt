package ai.onnxruntime.example.objectdetection.detection

/**
 * Detector listener that includes performance monitoring
 */
class PerformanceMonitoringListener(
    private val delegate: DetectorListener
) : DetectorListener {
    private var totalInferenceTime = 0L
    private var inferenceCount = 0
    private var errors = 0

    override fun onResults(
        result: ai.onnxruntime.example.objectdetection.processing.DetectionResult,
        inferenceTime: Long,
        imageHeight: Int,
        imageWidth: Int
    ) {
        synchronized(this) {
            totalInferenceTime += inferenceTime
            inferenceCount++
        }

        delegate.onResults(result, inferenceTime, imageHeight, imageWidth)
    }

    override fun onError(error: String) {
        synchronized(this) {
            errors++
        }

        delegate.onError(error)
    }

    fun getAverageInferenceTime(): Float {
        return if (inferenceCount > 0) {
            totalInferenceTime.toFloat() / inferenceCount
        } else {
            0f
        }
    }

    fun getErrorRate(): Float {
        val total = inferenceCount + errors
        return if (total > 0) {
            errors.toFloat() / total
        } else {
            0f
        }
    }

    fun reset() {
        synchronized(this) {
            totalInferenceTime = 0
            inferenceCount = 0
            errors = 0
        }
    }
}
