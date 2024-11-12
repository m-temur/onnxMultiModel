package ai.onnxruntime.example.objectdetection.detection

import ai.onnxruntime.example.objectdetection.models.DetectionModel
import ai.onnxruntime.example.objectdetection.processing.DetectionResult
import ai.onnxruntime.example.objectdetection.processing.PadimProcessing
import ai.onnxruntime.example.objectdetection.processing.PatchcoreProcessing
import ai.onnxruntime.example.objectdetection.processing.ProcessingStrategy
import ai.onnxruntime.example.objectdetection.processing.YoloProcessing
import ai.onnxruntime.example.objectdetection.runtime.OnnxRuntimeWrapper
import ai.onnxruntime.example.objectdetection.utils.BitmapUtils
import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log

class AnomalyDetector(
    private val context: Context,
    private val detectorListener: DetectorListener
) {
    private var onnxRuntime: OnnxRuntimeWrapper? = null
    private var processingStrategy: ProcessingStrategy? = null
    private var isInitialized = false
    private var currentModel: DetectionModel? = null

    companion object {
        private const val TAG = "AnomalyDetector"
    }

    /**
     * Initializes the detector with specified model
     */
    fun initialize(model: DetectionModel) {
        try {
            // Cleanup any existing resources
            close()

            // Create new processing strategy based on model type
            processingStrategy = when (model) {
                is DetectionModel.Patchcore -> {
                    onnxRuntime = OnnxRuntimeWrapper(context, "patchcore.onnx")
                    PatchcoreProcessing(model)
                }
                is DetectionModel.Padim -> {
                    onnxRuntime = OnnxRuntimeWrapper(context, "padim.onnx")
                    PadimProcessing(model)
                }
                is DetectionModel.Yolo -> {
                    onnxRuntime = OnnxRuntimeWrapper(context, "yolo.onnx")
                    YoloProcessing(model)
                }
            }

            // Initialize ONNX Runtime
            onnxRuntime?.initialize()

            currentModel = model
            isInitialized = true

            Log.d(TAG, "Initialized detector with model: ${model::class.simpleName}")
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing detector: ${e.message}")
            isInitialized = false
            throw e
        }
    }

    /**
     * Performs detection on the provided bitmap
     */
    fun detect(bitmap: Bitmap, imageRotation: Int) {
        if (!isInitialized) {
            detectorListener.onError("Detector not initialized")
            return
        }

        try {
            val inferenceTime = SystemClock.uptimeMillis()
            val result = processImageAndRunInference(bitmap, imageRotation)

            detectorListener.onResults(
                result = result,
                inferenceTime = SystemClock.uptimeMillis() - inferenceTime,
                imageHeight = bitmap.height,
                imageWidth = bitmap.width
            )
        } catch (e: Exception) {
            Log.e(TAG, "Detection failed: ${e.message}")
            detectorListener.onError("Detection failed: ${e.message}")
        }
    }

    /**
     * Processes image and runs inference
     */
    private fun processImageAndRunInference(bitmap: Bitmap, rotation: Int): DetectionResult {
        // Handle rotation first
        val rotatedBitmap = if (rotation != 0) {
            BitmapUtils.rotateBitmap(bitmap, rotation)
        } else {
            bitmap
        }

        try {
            // Preprocess the image
            val inputTensor = processingStrategy?.preprocess(rotatedBitmap)
                ?: throw IllegalStateException("Processing strategy not initialized")

            // Run inference
            val outputs = onnxRuntime?.run(mapOf("input" to inputTensor))
                ?: throw IllegalStateException("Failed to run inference")

            // Post-process results
            return processingStrategy?.postprocess(outputs)
                ?: throw IllegalStateException("Failed to process results")
        } finally {
            // Clean up rotated bitmap if it's different from input
            if (rotation != 0) {
                rotatedBitmap.recycle()
            }
        }
    }

    /**
     * Visualizes the detection result
     */
    fun visualizeResult(result: DetectionResult): Bitmap? {
        if (!isInitialized) return null

        return try {
            processingStrategy?.visualizeResult(
                originalBitmap = getCurrentPreviewFrame() ?: return null,
                result = result
            )
        } catch (e: Exception) {
            Log.e(TAG, "Error visualizing result: ${e.message}")
            null
        }
    }

    /**
     * Gets the current preview frame (if available)
     */
    private fun getCurrentPreviewFrame(): Bitmap? {
        // Implement this based on your camera preview implementation
        // This could return the last processed frame or get a new frame from the camera
        return null
    }

    /**
     * Releases resources
     */
    fun close() {
        try {
            onnxRuntime?.close()
            onnxRuntime = null
            processingStrategy = null
            currentModel = null
            isInitialized = false
        } catch (e: Exception) {
            Log.e(TAG, "Error closing detector: ${e.message}")
        }
    }

    /**
     * Checks if detector is initialized
     */
    fun isInitialized(): Boolean = isInitialized

    /**
     * Gets current model information
     */
    fun getCurrentModel(): DetectionModel? = currentModel

    /**
     * Gets input size for current model
     */
    fun getInputSize(): Pair<Int, Int>? {
        return when (val model = currentModel) {
            is DetectionModel.Patchcore -> model.inputSize
            is DetectionModel.Padim -> model.inputSize
            is DetectionModel.Yolo -> model.inputSize
            null -> null
        }
    }

    /**
     * Class to build detector with configuration
     */
    class Builder(private val context: Context) {
        private var detectorListener: DetectorListener? = null
        private var model: DetectionModel? = null
        private var enableLogging: Boolean = false

        fun setDetectorListener(listener: DetectorListener) = apply {
            this.detectorListener = listener
        }

        fun setModel(model: DetectionModel) = apply {
            this.model = model
        }

        fun enableLogging(enable: Boolean) = apply {
            this.enableLogging = enable
        }

        fun build(): AnomalyDetector {
            requireNotNull(detectorListener) { "DetectorListener must be set" }
            requireNotNull(model) { "Model must be set" }

            return AnomalyDetector(context, detectorListener!!).also { detector ->
                if (enableLogging) {
                    Log.d(TAG, "Building detector with model: ${model!!::class.simpleName}")
                }
                detector.initialize(model!!)
            }
        }
    }
}