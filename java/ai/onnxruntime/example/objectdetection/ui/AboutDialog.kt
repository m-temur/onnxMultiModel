package ai.onnxruntime.example.objectdetection.ui


import ai.onnxruntime.example.objectdetection.databinding.DialogAboutBinding
import ai.onnxruntime.example.objectdetection.databinding.DialogModelInfoBinding
import ai.onnxruntime.example.objectdetection.models.DetectionModel
import android.app.Dialog
import android.os.Bundle
import android.view.LayoutInflater
import androidx.appcompat.app.AlertDialog
import androidx.fragment.app.DialogFragment

class AboutDialog : DialogFragment() {
    private var _binding: DialogAboutBinding? = null
    private val binding get() = _binding!!

    override fun onCreateDialog(savedInstanceState: Bundle?): Dialog {
        _binding = DialogAboutBinding.inflate(LayoutInflater.from(context))

        return AlertDialog.Builder(requireContext())
            .setView(binding.root)
            .setPositiveButton("OK", null)
            .create()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}

class ModelInfoDialog : DialogFragment() {
    private var _binding: DialogModelInfoBinding? = null
    private val binding get() = _binding!!

    private var currentModel: DetectionModel? = null
    private var inferenceTime: Float = 0f
    private var errorRate: Float = 0f

    companion object {
        fun newInstance(
            model: DetectionModel,
            inferenceTime: Float,
            errorRate: Float
        ): ModelInfoDialog {
            return ModelInfoDialog().apply {
                this.currentModel = model
                this.inferenceTime = inferenceTime
                this.errorRate = errorRate
            }
        }
    }

    override fun onCreateDialog(savedInstanceState: Bundle?): Dialog {
        _binding = DialogModelInfoBinding.inflate(LayoutInflater.from(context))

        updateModelInfo()

        return AlertDialog.Builder(requireContext())
            .setView(binding.root)
            .setPositiveButton("Close", null)
            .create()
    }

    private fun updateModelInfo() {
        currentModel?.let { model ->
            binding.apply {
                modelNameText.text = "Model: ${model::class.simpleName}"

                when (model) {
                    is DetectionModel.Patchcore -> {
                        modelDetailsText.text = """
                            Type: Patchcore
                            Input Size: ${model.inputSize.first}x${model.inputSize.second}
                            Threshold: ${model.threshold}
                        """.trimIndent()
                    }
                    is DetectionModel.Padim -> {
                        modelDetailsText.text = """
                            Type: Padim
                            Input Size: ${model.inputSize.first}x${model.inputSize.second}
                            Threshold: ${model.threshold}
                        """.trimIndent()
                    }
                    is DetectionModel.Yolo -> {
                        modelDetailsText.text = """
                            Type: YOLO
                            Input Size: ${model.inputSize.first}x${model.inputSize.second}
                            Confidence Threshold: ${model.confidenceThreshold}
                            NMS Threshold: ${model.nmsThreshold}
                        """.trimIndent()
                    }
                }

                performanceText.text = """
                    Average Inference Time: %.2f ms
                    Error Rate: %.2f%%
                """.trimIndent().format(inferenceTime, errorRate * 100)
            }
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}

class ErrorDialog : DialogFragment() {
    companion object {
        private const val ARG_MESSAGE = "error_message"

        fun newInstance(message: String): ErrorDialog {
            return ErrorDialog().apply {
                arguments = Bundle().apply {
                    putString(ARG_MESSAGE, message)
                }
            }
        }
    }

    override fun onCreateDialog(savedInstanceState: Bundle?): Dialog {
        val message = arguments?.getString(ARG_MESSAGE) ?: "An error occurred"

        return AlertDialog.Builder(requireContext())
            .setTitle("Error")
            .setMessage(message)
            .setPositiveButton("OK", null)
            .create()
    }
}