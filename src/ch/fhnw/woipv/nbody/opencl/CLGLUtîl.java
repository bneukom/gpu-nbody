package ch.fhnw.woipv.nbody.opencl;

import static org.jocl.CL.*;
import jogamp.opengl.GLContextImpl;
import jogamp.opengl.GLDrawableImpl;
import jogamp.opengl.egl.EGLContext;
import jogamp.opengl.macosx.cgl.MacOSXCGLContext;
import jogamp.opengl.windows.wgl.WindowsWGLContext;
import jogamp.opengl.x11.glx.X11GLXContext;

import org.jocl.cl_context_properties;

import com.jogamp.nativewindow.NativeSurface;
import com.jogamp.opengl.GL;
import com.jogamp.opengl.GLContext;

public class CLGLUtîl {
	/**
	 * Initializes the given context properties so that they may be used to
	 * create an OpenCL context for the given GL object.
	 * 
	 * @param contextProperties
	 *            The context properties
	 * @param gl
	 *            The GL object
	 */
	public static void initContextProperties(final cl_context_properties contextProperties, final GL gl) {
		// Adapted from http://jogamp.org/jocl/www/

		final GLContext glContext = gl.getContext();
		if (!glContext.isCurrent()) {
			throw new IllegalArgumentException("OpenGL context is not current. This method should be called " + "from the OpenGL rendering thread, when the context is current.");
		}

		final long glContextHandle = glContext.getHandle();
		final GLContextImpl glContextImpl = (GLContextImpl) glContext;
		final GLDrawableImpl glDrawableImpl = glContextImpl.getDrawableImpl();
		final NativeSurface nativeSurface = glDrawableImpl.getNativeSurface();

		if (glContext instanceof X11GLXContext) {
			final long displayHandle = nativeSurface.getDisplayHandle();
			contextProperties.addProperty(CL_GL_CONTEXT_KHR, glContextHandle);
			contextProperties.addProperty(CL_GLX_DISPLAY_KHR, displayHandle);
		} else if (glContext instanceof WindowsWGLContext) {
			final long surfaceHandle = nativeSurface.getSurfaceHandle();
			contextProperties.addProperty(CL_GL_CONTEXT_KHR, glContextHandle);
			contextProperties.addProperty(CL_WGL_HDC_KHR, surfaceHandle);
		} else if (glContext instanceof MacOSXCGLContext) {
			contextProperties.addProperty(CL_CGL_SHAREGROUP_KHR, glContextHandle);
		} else if (glContext instanceof EGLContext) {
			final long displayHandle = nativeSurface.getDisplayHandle();
			contextProperties.addProperty(CL_GL_CONTEXT_KHR, glContextHandle);
			contextProperties.addProperty(CL_EGL_DISPLAY_KHR, displayHandle);
		} else {
			throw new RuntimeException("unsupported GLContext: " + glContext);
		}
	}
}
