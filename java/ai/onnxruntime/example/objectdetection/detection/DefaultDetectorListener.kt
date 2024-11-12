package ai.onnxruntime.example.objectdetection.detection

/**
 * Default implementation of DetectorListener that can be used for testing or as a base class
 */
open class DefaultDetectorListener : DetectorListener {
    override fun onResults(
        result: ai.onnxruntime.example.objectdetection.processing.DetectionResult,
        inferenceTime: Long,
        imageHeight: Int,
        imageWidth: Int
    ) {
        // Default empty implementation
    }

    override fun onError(error: String) {
        // Default empty implementation
    }
}