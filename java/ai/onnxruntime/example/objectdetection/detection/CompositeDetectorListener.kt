package ai.onnxruntime.example.objectdetection.detection

/**
 * Composite detector listener that can notify multiple listeners
 */
class CompositeDetectorListener : DetectorListener {
    private val listeners = mutableListOf<DetectorListener>()

    fun addListener(listener: DetectorListener) {
        synchronized(listeners) {
            listeners.add(listener)
        }
    }

    fun removeListener(listener: DetectorListener) {
        synchronized(listeners) {
            listeners.remove(listener)
        }
    }

    override fun onResults(
        result: ai.onnxruntime.example.objectdetection.processing.DetectionResult,
        inferenceTime: Long,
        imageHeight: Int,
        imageWidth: Int
    ) {
        synchronized(listeners) {
            listeners.forEach { listener ->
                listener.onResults(result, inferenceTime, imageHeight, imageWidth)
            }
        }
    }

    override fun onError(error: String) {
        synchronized(listeners) {
            listeners.forEach { listener ->
                listener.onError(error)
            }
        }
    }
}
