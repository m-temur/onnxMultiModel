package ai.onnxruntime.example.objectdetection.detection

/**
 * Interface for handling detection results and errors
 */
interface DetectorListener {
    /**
     * Called when detection is completed successfully
     *
     * @param result The detection result containing score, label, and visualization data
     * @param inferenceTime Time taken for inference in milliseconds
     * @param imageHeight Original image height
     * @param imageWidth Original image width
     */
    fun onResults(
        result: ai.onnxruntime.example.objectdetection.processing.DetectionResult,
        inferenceTime: Long,
        imageHeight: Int,
        imageWidth: Int
    )

    /**
     * Called when an error occurs during detection
     *
     * @param error Error message describing what went wrong
     */
    fun onError(error: String)
}