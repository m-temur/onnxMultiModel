package ai.onnxruntime.example.objectdetection.ui

import ai.onnxruntime.example.objectdetection.R
import ai.onnxruntime.example.objectdetection.databinding.ActivityMainBinding
import ai.onnxruntime.example.objectdetection.models.DetectionModel
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import android.util.Log

import android.view.Menu
import android.view.MenuItem
import android.widget.Toast
import androidx.fragment.app.Fragment

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    companion object {
        private const val TAG = "MainActivity"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Setup toolbar
        setupToolbar()

        // Load initial fragment if this is first creation
        if (savedInstanceState == null) {
            loadFragment(CameraFragment())
        }
    }

    private fun setupToolbar() {
        binding.toolbar?.let { toolbar ->
            setSupportActionBar(toolbar)
            supportActionBar?.setDisplayHomeAsUpEnabled(false)
            supportActionBar?.title = getString(R.string.app_name)
        }
    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menuInflater.inflate(R.menu.main_menu, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            R.id.action_about -> {
                showAboutDialog()
                true
            }
            R.id.action_model_info -> {
                showModelInfo()
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }

    private fun showAboutDialog() {
        AboutDialog().show(supportFragmentManager, "about_dialog")
    }

    private fun showModelInfo() {
        val currentFragment = supportFragmentManager.findFragmentById(R.id.fragment_container)
        if (currentFragment is CameraFragment) {
            currentFragment.getCurrentModelInfo()?.let { (model, inferenceTime, errorRate) ->
                ModelInfoDialog.newInstance(model, inferenceTime, errorRate)
                    .show(supportFragmentManager, "model_info_dialog")
            } ?: showError("Model information not available")
        }
    }

    private fun showError(message: String) {
        ErrorDialog.newInstance(message)
            .show(supportFragmentManager, "error_dialog")
    }

    private fun loadFragment(fragment: Fragment) {
        try {
            supportFragmentManager.beginTransaction()
                .replace(R.id.fragment_container, fragment)
                .commitAllowingStateLoss()
        } catch (e: Exception) {
            Log.e(TAG, "Error loading fragment: ${e.message}")
            showError(getString(R.string.error_loading_camera))
        }
    }

    override fun onDestroy() {
        // Clean up resources
        try {
            supportFragmentManager.fragments.forEach { fragment ->
                supportFragmentManager.beginTransaction()
                    .remove(fragment)
                    .commitAllowingStateLoss()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error cleaning up fragments: ${e.message}")
        }
        super.onDestroy()
    }
}