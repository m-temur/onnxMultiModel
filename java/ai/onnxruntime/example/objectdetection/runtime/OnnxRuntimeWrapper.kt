package ai.onnxruntime.example.objectdetection.runtime

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log

class OnnxRuntimeWrapper(
    private val context: Context,
    private val modelPath: String
) {
    private var ortEnvironment: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private var isInitialized = false

    companion object {
        private const val TAG = "OnnxRuntimeWrapper"
    }

    fun initialize() {
        try {
            ortEnvironment = OrtEnvironment.getEnvironment()
            val modelBytes = context.assets.open(modelPath).use { it.readBytes() }
            val sessionOptions = OrtSession.SessionOptions()
            ortSession = ortEnvironment?.createSession(modelBytes, sessionOptions)
            isInitialized = true

            // Log successful initialization
            Log.d(TAG, "Initialized ONNX Runtime with model: $modelPath")
            logModelInfo()
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing ONNX Runtime: ${e.message}")
            isInitialized = false
            throw e
        }
    }

    fun run(inputs: Map<String, OnnxTensor>): Map<String, OnnxTensor> {
        if (!isInitialized) {
            throw IllegalStateException("ONNX Runtime not initialized")
        }

        try {
            val result: OrtSession.Result? = ortSession?.run(inputs)
            val outputMap: MutableMap<String, OnnxTensor> = mutableMapOf()

            result?.let { sessionResult ->
                ortSession?.outputInfo?.keys?.forEach { outputName ->
                    try {
                        val tensor = sessionResult.get(outputName) as? OnnxTensor
                        if (tensor != null) {
                            outputMap[outputName] = tensor
                        } else {
                            Log.w(TAG, "Output tensor '$outputName' is null or not an OnnxTensor")
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Error getting output tensor '$outputName': ${e.message}")
                    }
                }
            }

            return outputMap
        } catch (e: Exception) {
            Log.e(TAG, "Error running inference: ${e.message}")
            throw e
        }
    }

    fun close() {
        try {
            ortSession?.close()
            ortEnvironment?.close()
            ortSession = null
            ortEnvironment = null
            isInitialized = false
        } catch (e: Exception) {
            Log.e(TAG, "Error closing ONNX Runtime: ${e.message}")
        }
    }

    fun getInputNames(): List<String> {
        return ortSession?.inputInfo?.keys?.toList() ?: emptyList()
    }

    fun getOutputNames(): List<String> {
        return ortSession?.outputInfo?.keys?.toList() ?: emptyList()
    }

    fun logModelInfo() {
        try {
            Log.d(TAG, "Model Information:")
            Log.d(TAG, "Input Info:")
            ortSession?.inputInfo?.forEach { (name, _) ->
                Log.d(TAG, "  Input '$name'")
            }

            Log.d(TAG, "Output Info:")
            ortSession?.outputInfo?.forEach { (name, _) ->
                Log.d(TAG, "  Output '$name'")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error logging model info: ${e.message}")
        }
    }

    fun isInitialized(): Boolean = isInitialized
}