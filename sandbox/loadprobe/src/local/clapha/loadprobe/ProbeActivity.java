package local.clapha.loadprobe;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;

/**
 * Sandbox gate 1: does the game's own native stack load on real ARM64 Android?
 *
 * Loads the build's libraries one at a time, in the order cr-native-sandbox's ARM64 bring-up
 * used, and logs each result under tag CLAPHA_PROBE. Their emulated attempt SIGSEGV'd inside
 * libscid_sdk's initialisation; a native crash here shows in logcat with library + offset.
 * No network permission: nothing loaded here can reach any server.
 */
public class ProbeActivity extends Activity {
    private static final String TAG = "CLAPHA_PROBE";
    private static final String DEFAULT_ORDER =
            "c++_shared,fmod,fmodstudio,sentry,sentry-android,scid_sdk,g";

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        final String libs = getIntent().getStringExtra("libs");
        new Thread(() -> run(libs == null ? DEFAULT_ORDER : libs)).start();
    }

    private void run(String libs) {
        Log.i(TAG, "start order=" + libs + " pid=" + android.os.Process.myPid());
        for (String lib : libs.split(",")) {
            Log.i(TAG, "loading lib" + lib + ".so");
            long started = System.nanoTime();
            try {
                System.loadLibrary(lib);
                Log.i(TAG, "OK lib" + lib + ".so in " + (System.nanoTime() - started) / 1000000 + " ms");
            } catch (Throwable error) {
                Log.e(TAG, "FAIL lib" + lib + ".so: " + error);
            }
        }
        Log.i(TAG, "done");
    }
}
