package ai.onnxruntime.example.objectdetection.ui

import ai.onnxruntime.example.objectdetection.BuildConfig
import ai.onnxruntime.example.objectdetection.R
import ai.onnxruntime.example.objectdetection.databinding.FragmentCameraBinding
import ai.onnxruntime.example.objectdetection.detection.AnomalyDetector
import ai.onnxruntime.example.objectdetection.detection.CompositeDetectorListener
import ai.onnxruntime.example.objectdetection.detection.DetectorListener
import ai.onnxruntime.example.objectdetection.detection.PerformanceMonitoringListener
import ai.onnxruntime.example.objectdetection.detection.ResultStorageListener
import ai.onnxruntime.example.objectdetection.models.DetectionModel
import ai.onnxruntime.example.objectdetection.models.DetectionResult.Companion.LABEL_ANOMALOUS
import ai.onnxruntime.example.objectdetection.processing.DetectionResult
import ai.onnxruntime.example.objectdetection.utils.BitmapUtils
import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.view.animation.AccelerateDecelerateInterpolator
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.constraintlayout.widget.ConstraintLayout
import androidx.core.content.ContextCompat
import androidx.core.graphics.ColorUtils
import androidx.fragment.app.Fragment
import kotlinx.android.synthetic.main.fragment_camera.view.performanceTextView
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors


class CameraFragment : Fragment(), DetectorListener {
    private var _binding: FragmentCameraBinding? = null
    private val binding get() = _binding!!

    private var anomalyDetector: AnomalyDetector? = null
    private var currentDetectionModel: DetectionModel = DetectionModel.Patchcore()

    private val performanceListener = PerformanceMonitoringListener(this)
    private val resultStorage = ResultStorageListener()
    private val compositeListener = CompositeDetectorListener()

    private var cameraExecutor: ExecutorService? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var isProcessingImage = false
    private var currentResult: DetectionResult? = null
    private var isDetectorInitialized = false

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentCameraBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // Set up composite listener
        compositeListener.apply {
            addListener(performanceListener)
            addListener(resultStorage)
        }

        // Initialize camera executor
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Initialize detector
        initializeDetector()

        // Setup model selection spinner
        setupModelSelection()

        // Start camera if permissions are granted
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            requestPermissions(REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }
    }

    private fun setupModelSelection() {
        val models = listOf("Patchcore", "Padim", "YOLO")
        binding.modelSpinner.adapter = ArrayAdapter(
            requireContext(),
            android.R.layout.simple_spinner_dropdown_item,
            models
        )

        binding.modelSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                val newModel = when (position) {
                    0 -> DetectionModel.Patchcore()
                    1 -> DetectionModel.Padim()
                    2 -> DetectionModel.Yolo()
                    else -> return
                }

                if (newModel::class != currentDetectionModel::class) {
                    currentDetectionModel = newModel
                    anomalyDetector?.close()
                    initializeDetector()
                }
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext())

        cameraProviderFuture.addListener({
            try {
                cameraProvider = cameraProviderFuture.get()
                bindCameraUseCases()
            } catch (e: Exception) {
                Log.e(TAG, "Failed to get camera provider: ${e.message}")
                showToast("Failed to start camera")
            }
        }, ContextCompat.getMainExecutor(requireContext()))
    }

    private fun bindCameraUseCases() {
        try {
            val cameraProvider = cameraProvider ?: throw IllegalStateException("Camera initialization failed.")

            // Preview use case
            val preview = Preview.Builder()
                .setTargetRotation(binding.viewFinder.display.rotation)
                .build()

            // Image analysis use case
            imageAnalyzer = ImageAnalysis.Builder()
                .setTargetRotation(binding.viewFinder.display.rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build()
                .also { analysis ->
                    cameraExecutor?.let {
                        analysis.setAnalyzer(it) { imageProxy ->
                            try {
                                processImage(imageProxy)
                            } catch (e: Exception) {
                                Log.e(TAG, "Error processing image: ${e.message}")
                            } finally {
                                imageProxy.close()
                            }
                        }
                    }
                }

            try {
                // Must unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                camera = cameraProvider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageAnalyzer
                )

                // Attach the preview to PreviewView
                preview.setSurfaceProvider(binding.viewFinder.surfaceProvider)

            } catch (e: Exception) {
                Log.e(TAG, "Use case binding failed: ${e.message}")
                showToast("Camera initialization failed. Please restart the app.")
            }

        } catch (e: Exception) {
            Log.e(TAG, "Camera setup failed: ${e.message}")
        }
    }

    private fun processImage(imageProxy: ImageProxy) {
        if (isProcessingImage) {
            return
        }

        try {
            isProcessingImage = true

            // Clean up previous result
            currentResult?.let {
                // Any cleanup needed for the previous result
            }
            currentResult = null

            // Use the new method with target size
            val bitmap = BitmapUtils.imageProxyToBitmapWithResize(
                imageProxy = imageProxy,
                targetSize = TARGET_SIZE // This should match your model's input size
            )

            bitmap?.let {
                try {
                    anomalyDetector?.detect(it, imageProxy.imageInfo.rotationDegrees)
                } finally {
                    bitmap.recycle()
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error processing image: ${e.message}")
        } finally {
            isProcessingImage = false
        }
    }

    companion object {
        private const val TAG = "CameraFragment"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
        private const val TARGET_SIZE = 224 // Your model's input size
    }

    private fun initializeDetector() {
        try {
            anomalyDetector = AnomalyDetector.Builder(requireContext())
                .setDetectorListener(this)
                .setModel(currentDetectionModel)
                .enableLogging(BuildConfig.DEBUG)
                .build()
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing detector: ${e.message}")
            showToast("Failed to initialize detector")
        }
    }

    // Switch models
    private fun switchModel(newModel: DetectionModel) {
        try {
            anomalyDetector?.close()
            currentDetectionModel = newModel
            initializeDetector()
        } catch (e: Exception) {
            Log.e(TAG, "Error switching models: ${e.message}")
            showToast("Failed to switch models")
        }
    }

    override fun onResults(
        result: DetectionResult,
        inferenceTime: Long,
        imageHeight: Int,
        imageWidth: Int
    ) {
        if (!isAdded) return

        activity?.runOnUiThread {
            try {
                // Store the current result
                currentResult = result

                // Update the output image with visualization
                binding.outputImageView.setImageBitmap(
                    anomalyDetector?.visualizeResult(result)
                )

                // Update performance metrics
                val avgInferenceTime = performanceListener.getAverageInferenceTime()
                val errorRate = performanceListener.getErrorRate()

                // Update metrics display
                updateResultsDisplay(
                    result = result,
                    inferenceTime = inferenceTime,
                    avgInferenceTime = avgInferenceTime,
                    errorRate = errorRate
                )

                // Update UI based on detection result
                updateUIBasedOnResult(result)

            } catch (e: Exception) {
                Log.e(TAG, "Error updating UI with results: ${e.message}")
                e.printStackTrace()
            }
        }
    }

    private fun updateResultsDisplay(
        result: DetectionResult,
        inferenceTime: Long,
        avgInferenceTime: Float,
        errorRate: Float
    ) {
        val resultColor = if (result.label == LABEL_ANOMALOUS) {
            ContextCompat.getColor(requireContext(), android.R.color.holo_red_light)
        } else {
            ContextCompat.getColor(requireContext(), android.R.color.holo_green_light)
        }

        binding.performanceTextView.apply {
            setBackgroundColor(ColorUtils.setAlphaComponent(resultColor, 128))
            text = buildString {
                append("Model: ${currentDetectionModel::class.simpleName}\n")
                append(String.format("Score: %.3f\n", result.score))
                append("Label: ${result.label}\n")
                append(String.format("Confidence: %.1f%%\n", result.confidence * 100))
                append(String.format("Inference: %d ms\n", inferenceTime))
                append(String.format("Avg. Inference: %.1f ms\n", avgInferenceTime))
                append(String.format("Error Rate: %.2f%%", errorRate * 100))
            }
        }

        // Optionally animate the text view when anomaly is detected
        if (result.label == LABEL_ANOMALOUS) {
            animateAnomalyDetection()
        }
    }

    private fun updateUIBasedOnResult(result: DetectionResult) {
        // Update background color based on detection result
        binding.outputImageView.setBackgroundColor(
            if (result.label == LABEL_ANOMALOUS) {
                ColorUtils.setAlphaComponent(Color.RED, 32)
            } else {
                Color.TRANSPARENT
            }
        )

        // Update anomaly status if available
        binding.anomalyStatusView?.apply {
            visibility = if (result.label == LABEL_ANOMALOUS) {
                View.VISIBLE
            } else {
                View.GONE
            }
            text = if (result.label == LABEL_ANOMALOUS) {
                "Anomaly Detected!"
            } else {
                ""
            }
        }
    }

    private fun animateAnomalyDetection() {
        binding.performanceTextView.apply {
            alpha = 1.0f
            animate()
                .alpha(0.6f)
                .setDuration(300)
                .setInterpolator(AccelerateDecelerateInterpolator())
                .withEndAction {
                    animate()
                        .alpha(1.0f)
                        .setDuration(300)
                        .setInterpolator(AccelerateDecelerateInterpolator())
                        .start()
                }
                .start()
        }
    }
    // Add this to your layout file
    private fun setupAdditionalViews() {
        // Example of adding a confidence indicator
        binding.root.apply {
            // Add confidence indicator
            val confidenceIndicator = ProgressBar(context).apply {
                id = View.generateViewId()
                layoutParams = ConstraintLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT
                ).apply {
                    topToBottom = R.id.performanceTextView
                    startToStart = ConstraintLayout.LayoutParams.PARENT_ID
                    endToEnd = ConstraintLayout.LayoutParams.PARENT_ID
                    marginStart = 16
                    marginEnd = 16
                    topMargin = 8
                }
                max = 100
                progressDrawable = ContextCompat.getDrawable(
                    context,
                    android.R.drawable.progress_horizontal
                )
            }
            addView(confidenceIndicator)

            // Add anomaly status view
            val anomalyStatusView = TextView(context).apply {
                id = View.generateViewId()
                layoutParams = ConstraintLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT
                ).apply {
                    topToBottom = confidenceIndicator.id
                    startToStart = ConstraintLayout.LayoutParams.PARENT_ID
                    endToEnd = ConstraintLayout.LayoutParams.PARENT_ID
                    topMargin = 8
                }
                setTextColor(Color.RED)
                textSize = 18f
                visibility = View.GONE
            }
            addView(anomalyStatusView)
        }

        fun initializeDetector() {
            try {
                anomalyDetector = AnomalyDetector.Builder(requireContext())
                    .setDetectorListener(this)
                    .setModel(currentDetectionModel)
                    .enableLogging(BuildConfig.DEBUG)
                    .build()
                isDetectorInitialized = true  // Set to true when initialization succeeds
            } catch (e: Exception) {
                Log.e(TAG, "Error initializing detector: ${e.message}")
                showToast("Failed to initialize detector")
                isDetectorInitialized = false  // Set to false if initialization fails
            }
        }

        fun getCurrentModelInfo(): Triple<DetectionModel, Float, Float>? {
            return if (isDetectorInitialized && anomalyDetector != null) {
                Triple(
                    currentDetectionModel,
                    performanceListener.getAverageInferenceTime(),
                    performanceListener.getErrorRate()
                )
            } else {
                null
            }
        }

    }


    override fun onError(error: String) {
        if (!isAdded) return

        activity?.runOnUiThread {
            showToast(error)
        }
    }

    private fun updatePerformanceMetrics(
        avgInferenceTime: Float,
        errorRate: Float,
        result: DetectionResult
    ) {
        binding.performanceTextView.text = String.format(
            "Model: %s\nScore: %.3f\nLabel: %s\nAvg. Inference: %.1f ms\nError Rate: %.2f%%",
            currentDetectionModel::class.simpleName,
            result.score,
            result.label,
            avgInferenceTime,
            errorRate * 100
        )
    }

    private fun showToast(message: String) {
        activity?.runOnUiThread {
            Toast.makeText(context, message, Toast.LENGTH_SHORT).show()
        }
    }

    override fun onPause() {
        super.onPause()
        try {
            cameraProvider?.unbindAll()
        } catch (e: Exception) {
            Log.e(TAG, "Error unbinding camera uses cases: ${e.message}")
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        try {
            compositeListener.removeListener(performanceListener)
            compositeListener.removeListener(resultStorage)
            performanceListener.reset()
            resultStorage.clearResults()

            currentResult?.let {
                // Any cleanup needed
            }
            currentResult = null

            cameraExecutor?.shutdown()
            cameraExecutor = null

            anomalyDetector?.close()
            anomalyDetector = null

            _binding = null
        } catch (e: Exception) {
            Log.e(TAG, "Error cleaning up resources: ${e.message}")
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(requireContext(), it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                showToast("Permissions not granted.")
                requireActivity().finish()
            }
        }
    }

}