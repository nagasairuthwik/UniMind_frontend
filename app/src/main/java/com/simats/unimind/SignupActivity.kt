package com.simats.unimind

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.text.SpannableString
import android.text.Spanned
import android.text.TextPaint
import android.text.method.LinkMovementMethod
import android.text.style.ClickableSpan
import android.text.style.StyleSpan
import android.view.View
import android.view.inputmethod.InputMethodManager
import android.widget.Button
import android.widget.EditText
import android.widget.ImageButton
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.appcompat.widget.PopupMenu
import androidx.core.content.ContextCompat
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response

class SignupActivity : ComponentActivity() {

    private var passwordVisible = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_signup)

        // Dismiss keyboard when tapping outside edit fields
        fun dismissKeyboardOnTouch() {
            currentFocus?.clearFocus()
            hideKeyboard()
        }
        findViewById<View>(android.R.id.content).setOnTouchListener { _, _ ->
            dismissKeyboardOnTouch()
            false
        }
        findViewById<View>(R.id.signup_scroll).setOnTouchListener { _, _ ->
            dismissKeyboardOnTouch()
            false
        }

        findViewById<ImageButton>(R.id.signup_back).setOnClickListener {
            hideKeyboard()
            currentFocus?.clearFocus()
            finish()
        }

        findViewById<ImageButton>(R.id.signup_menu).setOnClickListener { v ->
            PopupMenu(this, v).apply {
                menu.add(0, 1, 0, getString(R.string.menu_sign_in))
                menu.add(0, 2, 1, getString(R.string.menu_help))
                setOnMenuItemClickListener { item ->
                    when (item.itemId) {
                        1 -> { startActivity(Intent(this@SignupActivity, SignInActivity::class.java)); finish() }
                        2 -> startActivity(Intent(this@SignupActivity, HelpSupportActivity::class.java))
                        else -> { }
                    }
                    true
                }
                show()
            }
        }

        val fullNameEdit = findViewById<EditText>(R.id.signup_full_name)
        val emailEdit = findViewById<EditText>(R.id.signup_email)
        val passwordEdit = findViewById<EditText>(R.id.signup_password)
        val confirmPasswordEdit = findViewById<EditText>(R.id.signup_confirm_password)

        findViewById<ImageButton>(R.id.signup_password_toggle).setOnClickListener {
            passwordVisible = !passwordVisible
            passwordEdit.inputType = if (passwordVisible) {
                android.text.InputType.TYPE_CLASS_TEXT or android.text.InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD
            } else {
                android.text.InputType.TYPE_CLASS_TEXT or android.text.InputType.TYPE_TEXT_VARIATION_PASSWORD
            }
            findViewById<ImageButton>(R.id.signup_password_toggle).setImageResource(
                if (passwordVisible) R.drawable.ic_eye_off_outline else R.drawable.ic_eye_outline
            )
        }

        findViewById<Button>(R.id.signup_continue).setOnClickListener {
            hideKeyboard()
            currentFocus?.clearFocus()
            val fullName = fullNameEdit.text.toString().trim()
            val email = emailEdit.text.toString().trim()
            val password = passwordEdit.text.toString()
            val confirmPassword = confirmPasswordEdit.text.toString()

            if (fullName.isEmpty()) {
                fullNameEdit.error = "Enter your name"
                fullNameEdit.requestFocus()
                return@setOnClickListener
            }
            if (email.isEmpty()) {
                emailEdit.error = "Enter your email"
                emailEdit.requestFocus()
                return@setOnClickListener
            }
            if (password.isEmpty()) {
                passwordEdit.error = "Create a password"
                passwordEdit.requestFocus()
                return@setOnClickListener
            }
            if (!isStrongPassword(password)) {
                passwordEdit.error = getString(R.string.signup_password_requirement)
                passwordEdit.requestFocus()
                return@setOnClickListener
            }
            if (confirmPassword.isEmpty()) {
                confirmPasswordEdit.error = "Re-enter your password"
                confirmPasswordEdit.requestFocus()
                return@setOnClickListener
            }
            if (password != confirmPassword) {
                confirmPasswordEdit.error = getString(R.string.signup_password_mismatch_error)
                confirmPasswordEdit.requestFocus()
                return@setOnClickListener
            }

            // Call backend /signup via Retrofit (JSON body)
            ApiClient.service.signup(SignupRequest(fullName, email, password))
                .enqueue(object : Callback<okhttp3.ResponseBody> {
                    override fun onResponse(
                        call: Call<okhttp3.ResponseBody>,
                        response: Response<okhttp3.ResponseBody>
                    ) {
                        if (response.isSuccessful) {
                            hideKeyboard()
                            currentFocus?.clearFocus()
                            Toast.makeText(
                                this@SignupActivity,
                                "Signup successful",
                                Toast.LENGTH_SHORT
                            ).show()
                            // Immediately log the user in and start onboarding (Profile → All Set → Home)
                            loginAndStartOnboarding(fullName, email, password)
                        } else {
                            Toast.makeText(
                                this@SignupActivity,
                                "Signup failed: ${response.code()}",
                                Toast.LENGTH_SHORT
                            ).show()
                        }
                    }

                    override fun onFailure(
                        call: Call<okhttp3.ResponseBody>,
                        t: Throwable
                    ) {
                        Toast.makeText(
                            this@SignupActivity,
                            "Network error: ${t.message}",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                })
        }

        setupSignInLink()
    }

    /**
     * After successful signup, automatically log in the user and start onboarding.
     * Flow: Signup → (auto login) → ProfileSetupActivity → ... → AllSetActivity → SubscriptionActivity → MainActivity.
     */
    private fun loginAndStartOnboarding(fullName: String, email: String, password: String) {
        ApiClient.service.login(LoginRequest(email, password))
            .enqueue(object : Callback<LoginResponse> {
                override fun onResponse(
                    call: Call<LoginResponse>,
                    response: Response<LoginResponse>
                ) {
                    if (response.isSuccessful) {
                        val body = response.body()
                        val userId = body?.user?.id
                        if (userId != null && userId > 0) {
                            UserPrefs.saveUserId(this@SignupActivity, userId)
                            val displayName = body.user.full_name.takeIf { it.isNotBlank() } ?: fullName
                            UserPrefs.saveDisplayName(this@SignupActivity, displayName)
                            val intent = Intent(this@SignupActivity, ProfileSetupActivity::class.java)
                            intent.putExtra(ProfileSetupActivity.EXTRA_USER_ID, userId)
                            startActivity(intent)
                            finish()
                        } else {
                            // Fallback: go to SignIn so user can log in manually
                            val intent = Intent(this@SignupActivity, SignInActivity::class.java)
                            intent.putExtra(SignInActivity.EXTRA_PREFILL_EMAIL, email)
                            startActivity(intent)
                            finish()
                        }
                    } else {
                        // If login fails after signup, fall back to SignIn screen
                        val intent = Intent(this@SignupActivity, SignInActivity::class.java)
                        intent.putExtra(SignInActivity.EXTRA_PREFILL_EMAIL, email)
                        startActivity(intent)
                        finish()
                    }
                }

                override fun onFailure(call: Call<LoginResponse>, t: Throwable) {
                    // Network issue: let user log in on next screen
                    val intent = Intent(this@SignupActivity, SignInActivity::class.java)
                    intent.putExtra(SignInActivity.EXTRA_PREFILL_EMAIL, email)
                    startActivity(intent)
                    finish()
                }
            })
    }

    private fun setupSignInLink() {
        val signInView = findViewById<TextView>(R.id.signup_sign_in_link)
        val prefix = getString(R.string.signup_already_account)
        val signInText = getString(R.string.signup_sign_in)
        val fullText = "$prefix $signInText"

        val spannable = SpannableString(fullText)
        val signInStart = fullText.indexOf(signInText)
        if (signInStart >= 0) {
            val linkColor = ContextCompat.getColor(this, R.color.sign_in_link)
            spannable.setSpan(
                object : ClickableSpan() {
                    override fun onClick(widget: View) {
                        hideKeyboard()
                        currentFocus?.clearFocus()
                        startActivity(Intent(this@SignupActivity, SignInActivity::class.java))
                    }
                    override fun updateDrawState(ds: TextPaint) {
                        ds.color = linkColor
                        ds.isUnderlineText = false
                        ds.isFakeBoldText = true
                    }
                },
                signInStart,
                signInStart + signInText.length,
                Spanned.SPAN_EXCLUSIVE_EXCLUSIVE
            )
            spannable.setSpan(
                StyleSpan(android.graphics.Typeface.BOLD),
                signInStart,
                signInStart + signInText.length,
                Spanned.SPAN_EXCLUSIVE_EXCLUSIVE
            )
        }
        signInView.text = spannable
        signInView.movementMethod = LinkMovementMethod.getInstance()
        signInView.setTextColor(ContextCompat.getColor(this, R.color.signup_subtitle))
    }

    private fun hideKeyboard() {
        val imm = getSystemService(Context.INPUT_METHOD_SERVICE) as? InputMethodManager
        val token = currentFocus?.windowToken ?: window?.decorView?.windowToken
        if (token != null) {
            imm?.hideSoftInputFromWindow(token, 0)
        }
    }

    private fun isStrongPassword(password: String): Boolean {
        if (password.length < 8) return false
        val hasUpper = password.any { it.isUpperCase() }
        val hasLower = password.any { it.isLowerCase() }
        val hasDigit = password.any { it.isDigit() }
        val hasSpecial = password.any { !it.isLetterOrDigit() }
        return hasUpper && hasLower && hasDigit && hasSpecial
    }
}
