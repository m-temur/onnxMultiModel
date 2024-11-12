package ai.onnxruntime.example.objectdetection.models

// Represents different types of anomaly detection approaches
// Detection model data class
sealed class DetectionModel {
    data class Patchcore(
        val inputSize: Pair<Int, Int> = Pair(224, 224),
        val threshold: Float = IMAGE_THRESHOLD
    ) : DetectionModel() {
        companion object {
            private const val IMAGE_THRESHOLD = 46f
        }
    }

    data class Padim(
        val inputSize: Pair<Int, Int> = Pair(224, 224),
        val threshold: Float = 0.5f
    ) : DetectionModel()

    data class Yolo(
        val inputSize: Pair<Int, Int> = Pair(640, 640),
        val confidenceThreshold: Float = 0.5f,
        val nmsThreshold: Float = 0.4f
    ) : DetectionModel()
}

