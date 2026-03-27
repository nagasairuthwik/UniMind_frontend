package com.simats.unimind

import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.ResponseBody
import retrofit2.Call
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part
import retrofit2.http.Path
import java.util.concurrent.TimeUnit

/**
 * Single API client for UniMind app.
 * All backend calls go through ApiClient.service.
 */
data class SignupRequest(
    val full_name: String,
    val email: String,
    val password: String
)

data class LoginRequest(
    val email: String,
    val password: String
)

data class LoginUser(val id: Int, val email: String, val full_name: String)
data class LoginResponse(val success: Boolean, val message: String?, val user: LoginUser?)

data class ProfileSaveRequest(
    val user_id: Int,
    val full_name: String,
    val age: Int,
    val gender: String? = null,
    val avatar_url: String? = null,
    val goals: String? = null,
    val email: String? = null,
    val dob: String? = null,
    val phone: String? = null
)

data class UploadPhotoResponse(val success: Boolean, val avatar_url: String?, val message: String?)

data class GoalsUpdateRequest(
    val user_id: Int,
    val goals: String
)

data class PermissionsUpdateRequest(
    val user_id: Int,
    val allow_notifications: Boolean,
    val allow_location: Boolean,
    val allow_calendar: Boolean,
    val allow_health: Boolean
)

data class ProfileData(
    val user_id: Int,
    val full_name: String?,
    val email: String?,
    val age: Int?,
    val gender: String?,
    val avatar_url: String?,
    val goals: String?,
    val dob: String?,
    val phone: String?,
    val updated_at: String?
)

data class ProfileResponse(
    val success: Boolean,
    val message: String?,
    val profile: ProfileData?
)

/** Payload for POST /domain/finance: user data snapshot + optional AI suggestion. */
data class DomainFinanceRequest(
    val user_id: Int,
    val entry_date: String,
    val user_data: Map<String, Any?>,
    val ai_text: String? = null
)

/** Payload for POST /domain/lifestyle: lifestyle snapshot + optional AI suggestion. */
data class DomainLifestyleRequest(
    val user_id: Int,
    val entry_date: String,
    val user_data: Map<String, Any?>,
    val ai_text: String? = null
)

/** Payload for POST /domain/health: steps snapshot + optional AI suggestion. */
data class DomainHealthRequest(
    val user_id: Int,
    val entry_date: String,
    val user_data: Map<String, Any?>,
    val ai_text: String? = null
)

/** Payload for POST /domain/productivity: tasks snapshot + optional AI suggestion. */
data class DomainProductivityRequest(
    val user_id: Int,
    val entry_date: String,
    val user_data: Map<String, Any?>,
    val ai_text: String? = null
)

data class NotificationCreateRequest(
    val user_id: Int,
    val domain: String,
    val title: String,
    val body: String
)

data class NotificationItem(
    val id: Int,
    val user_id: Int,
    val domain: String,
    val title: String,
    val body: String,
    val is_read: Boolean,
    val created_at: String
)

data class NotificationsListResponse(
    val success: Boolean,
    val notifications: List<NotificationItem>,
    val count: Int
)

data class NotificationMarkReadRequest(
    val user_id: Int,
    val notification_id: Int? = null,
    val all: Boolean = false
)

data class SimpleResponse(
    val success: Boolean,
    val message: String?
)

data class ForgotPasswordOtpRequest(
    val email: String
)

data class ForgotPasswordResetRequest(
    val email: String,
    val otp: String,
    val new_password: String
)

data class ForgotPasswordVerifyRequest(
    val email: String,
    val otp: String
)

interface ApiService {

    @POST("signup")
    fun signup(@Body body: SignupRequest): Call<ResponseBody>

    @POST("login")
    fun login(@Body body: LoginRequest): Call<LoginResponse>

    @POST("profile")
    fun saveProfile(@Body body: ProfileSaveRequest): Call<ResponseBody>

    @Multipart
    @POST("profile/photo")
    fun uploadProfilePhoto(@Part photo: MultipartBody.Part): Call<UploadPhotoResponse>

    @POST("profile/goals")
    fun saveGoals(@Body body: GoalsUpdateRequest): Call<ResponseBody>

    @POST("permissions")
    fun savePermissions(@Body body: PermissionsUpdateRequest): Call<ResponseBody>

    @GET("profile/{user_id}")
    fun getProfile(@Path("user_id") userId: Int): Call<ProfileResponse>

    @POST("domain/finance")
    fun saveDomainFinance(@Body body: DomainFinanceRequest): Call<ResponseBody>

    @POST("domain/lifestyle")
    fun saveDomainLifestyle(@Body body: DomainLifestyleRequest): Call<ResponseBody>

    @POST("domain/health")
    fun saveDomainHealth(@Body body: DomainHealthRequest): Call<ResponseBody>

    @POST("domain/productivity")
    fun saveDomainProductivity(@Body body: DomainProductivityRequest): Call<ResponseBody>

    @POST("notifications")
    fun createNotification(@Body body: NotificationCreateRequest): Call<ResponseBody>

    @GET("notifications/{user_id}")
    fun getNotifications(@Path("user_id") userId: Int): Call<NotificationsListResponse>

    @POST("notifications/mark_read")
    fun markNotificationsRead(@Body body: NotificationMarkReadRequest): Call<ResponseBody>

    @POST("auth/forgot/send_otp")
    fun sendForgotOtp(@Body body: ForgotPasswordOtpRequest): Call<SimpleResponse>

    @POST("auth/forgot/verify_otp")
    fun verifyForgotOtp(@Body body: ForgotPasswordVerifyRequest): Call<SimpleResponse>

    @POST("auth/forgot/reset_password")
    fun resetForgotPassword(@Body body: ForgotPasswordResetRequest): Call<SimpleResponse>
}

object ApiClient {

    /**
     * Backend URL. Use your PC's IPv4 address so the phone can reach the server.
     * On PC: run "ipconfig" (Windows) and use the IPv4 Address (e.g. 192.168.1.5 or 10.0.0.5).
     * Phone and PC must be on the same Wi-Fi. URL must end with /
     */
    private const val BASE_URL = "http://10.221.3.106:5000"

    private val okHttpClient = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(15, TimeUnit.SECONDS)
        .writeTimeout(15, TimeUnit.SECONDS)
        .build()

    private val retrofit: Retrofit by lazy {
        Retrofit.Builder()
            .baseUrl(BASE_URL)
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
    }

    val service: ApiService by lazy {
        retrofit.create(ApiService::class.java)
    }
}

