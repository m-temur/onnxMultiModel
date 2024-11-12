package ai.onnxruntime.example.objectdetection.detection

/**
 * Implementation of DetectorListener that stores results for later retrieval
 */
class ResultStorageListener : DetectorListener {
    private val results = mutableListOf<DetectionMetadata>()
    private val maxStoredResults = 100

    override fun onResults(
        result: ai.onnxruntime.example.objectdetection.processing.DetectionResult,
        inferenceTime: Long,
        imageHeight: Int,
        imageWidth: Int
    ) {
        synchronized(results) {
            results.add(
                DetectionMetadata(
                    result = result,
                    inferenceTime = inferenceTime,
                    imageHeight = imageHeight,
                    imageWidth = imageWidth
                )
            )

            // Keep only the latest results
            if (results.size > maxStoredResults) {
                results.removeAt(0)
            }
        }
    }

    override fun onError(error: String) {
        // Log error or handle as needed
    }

    fun getResults(): List<DetectionMetadata> = results.toList()

    fun clearResults() {
        synchronized(results) {
            results.clear()
        }
    }

    /**
     * Wrapper class for detection results with additional metadata
     */
    data class DetectionMetadata(
        val result: ai.onnxruntime.example.objectdetection.processing.DetectionResult,
        val inferenceTime: Long,
        val imageHeight: Int,
        val imageWidth: Int,
        val timestamp: Long = System.currentTimeMillis()
    )
}