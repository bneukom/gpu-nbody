/*
 * JOCL - Java bindings for OpenCL
 * 
 * Copyright 2009 Marco Hutter - http://www.jocl.org/
 */

package ch.fhnw.woipv.nbody.visualization;

import static com.jogamp.opengl.GL.*;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.Frame;
import java.awt.Point;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionListener;
import java.awt.event.MouseWheelEvent;
import java.awt.event.MouseWheelListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.File;
import java.io.IOException;
import java.nio.IntBuffer;
import java.nio.file.Files;
import java.util.Arrays;

import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.SwingUtilities;

import org.jocl.Sizeof;

import ch.fhnw.woipv.nbody.simulation.GpuNBodySimulation;
import ch.fhnw.woipv.nbody.simulation.GpuNBodySimulation.Mode;
import ch.fhnw.woipv.nbody.simulation.NBodySimulation;
import ch.fhnw.woipv.nbody.simulation.universe.PlummerUniverseGenerator;
import ch.fhnw.woipv.nbody.simulation.universe.RandomCubicUniverseGenerator;

import com.jogamp.opengl.GL3;
import com.jogamp.opengl.GLAutoDrawable;
import com.jogamp.opengl.GLCapabilities;
import com.jogamp.opengl.GLEventListener;
import com.jogamp.opengl.GLProfile;
import com.jogamp.opengl.awt.GLCanvas;
import com.jogamp.opengl.util.Animator;

public class NBodyVisualizer implements GLEventListener {

	private NBodySimulation simulation;

	/**
	 * Whether the initialization method of this GLEventListener has already been called
	 */
	private boolean initialized = false;

	/**
	 * The vertex array object (required as of GL3)
	 */
	private int vertexArrayObject;

	/**
	 * The VBO identifier
	 */
	private int vertexBufferObject;

	/**
	 * The ID of the OpenGL shader program
	 */
	private int shaderProgramID;

	/**
	 * The translation in X-direction
	 */
	private float translationX = 0;

	/**
	 * The translation in Y-direction
	 */
	private float translationY = 0;

	/**
	 * The translation in Z-direction
	 */
	private float translationZ = -4;

	/**
	 * The rotation about the X-axis, in degrees
	 */
	private float rotationX = 40;

	/**
	 * The rotation about the Y-axis, in degrees
	 */
	private float rotationY = 30;

	/**
	 * The current projection matrix
	 */
	float projectionMatrix[] = new float[16];

	/**
	 * The current projection matrix
	 */
	float modelviewMatrix[] = new float[16];

	/**
	 * The animator
	 */
	private final Animator animator;

	/**
	 * The main frame of the application
	 */
	private final Frame frame;

	/**
	 * The OpenGL Canvas
	 */
	private GLCanvas glComponent;

	/**
	 * Inner class encapsulating the MouseMotionListener and MouseWheelListener for the interaction
	 */
	private class MouseControl implements MouseMotionListener, MouseWheelListener {
		private Point previousMousePosition = new Point();

		@Override
		public void mouseDragged(final MouseEvent e) {
			final int dx = e.getX() - previousMousePosition.x;
			final int dy = e.getY() - previousMousePosition.y;

			// If the left button is held down, move the object
			if ((e.getModifiersEx() & MouseEvent.BUTTON1_DOWN_MASK) == MouseEvent.BUTTON1_DOWN_MASK) {
				translationX += dx / 100.0f;
				translationY -= dy / 100.0f;
			}

			// If the right button is held down, rotate the object
			else if ((e.getModifiersEx() & MouseEvent.BUTTON3_DOWN_MASK) == MouseEvent.BUTTON3_DOWN_MASK) {
				rotationX += dx;
				rotationY += dy;
			}
			previousMousePosition = e.getPoint();
			updateModelviewMatrix();
		}

		@Override
		public void mouseMoved(final MouseEvent e) {
			previousMousePosition = e.getPoint();
		}

		@Override
		public void mouseWheelMoved(final MouseWheelEvent e) {
			// Translate along the Z-axis
			translationZ += e.getWheelRotation() * 0.25f;
			previousMousePosition = e.getPoint();
			updateModelviewMatrix();
		}
	}

	/**
	 * Creates a new JOCLSimpleGL3 sample.
	 * 
	 * @param capabilities
	 *            The GL capabilities
	 */
	public NBodyVisualizer(final GLCapabilities capabilities) {
		glComponent = new GLCanvas(capabilities);
		glComponent.setFocusable(true);
		glComponent.addGLEventListener(this);

		// Initialize the mouse and keyboard controls
		final MouseControl mouseControl = new MouseControl();
		glComponent.addMouseMotionListener(mouseControl);
		glComponent.addMouseWheelListener(mouseControl);
		updateModelviewMatrix();

		// Create and start an animator
		animator = new Animator(glComponent);
		animator.start();

		// Create the simulation
		simulation = new GpuNBodySimulation(Mode.GL_INTEROP, 2048 * 4, new RandomCubicUniverseGenerator(2));
		// simulation = new GpuNBodySimulation(Mode.GL_INTEROP, 2048 * 4, new PlummerUniverseGenerator());

		// Create the main frame
		frame = new JFrame("NBody Simulation");
		frame.addWindowListener(new WindowAdapter() {
			@Override
			public void windowClosing(final WindowEvent e) {
				runExit();
			}
		});
		frame.setLayout(new BorderLayout());
		glComponent.setPreferredSize(new Dimension(2048, 1280));
		frame.add(glComponent, BorderLayout.CENTER);
		frame.pack();
		frame.setVisible(true);
		frame.setLocationRelativeTo(null);
		glComponent.requestFocus();

	}

	/**
	 * Update the modelview matrix depending on the current translation and rotation
	 */
	private void updateModelviewMatrix() {
		final float m0[] = translation(translationX, translationY, translationZ);
		final float m1[] = rotationX(rotationX);
		final float m2[] = rotationY(rotationY);
		modelviewMatrix = multiply(multiply(m1, m2), m0);
	}

	/**
	 * Implementation of GLEventListener: Called to initialize the GLAutoDrawable
	 */
	@Override
	public void init(final GLAutoDrawable drawable) {
		// Perform the default GL initialization
		final GL3 gl = drawable.getGL().getGL3();

		gl.setSwapInterval(0);

		gl.glEnable(GL_DEPTH_TEST);
		gl.glPointSize(2);
		gl.glEnable(GL3.GL_PROGRAM_POINT_SIZE);
		gl.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

		// Initialize the shaders
		initShaders(gl);

		// Set up the viewport and projection matrix
		setupView(drawable);

		// Initialize the GL_ARB_vertex_buffer_object extension
		if (!gl.isExtensionAvailable("GL_ARB_vertex_buffer_object")) {

			new Thread(() -> {
				JOptionPane.showMessageDialog(null, "GL_ARB_vertex_buffer_object extension not available", "Unavailable extension", JOptionPane.ERROR_MESSAGE);
				runExit();
			}).start();
		}

		// initializes the simulation
		simulation.init(gl);

		// Initialize the OpenGL VBO and the OpenCL VBO memory object
		initVBOData(gl);

		initialized = true;
	}

	/**
	 * Initialize the shaders and the shader program
	 * 
	 * @param gl
	 *            The GL context
	 */
	private void initShaders(final GL3 gl) {
		try {
			final int vertexShaderID = gl.glCreateShader(GL3.GL_VERTEX_SHADER);
			gl.glShaderSource(vertexShaderID, 1, new String[] { readAllLines("shaders/simulation.vert") }, null);
			gl.glCompileShader(vertexShaderID);

			final int fragmentShaderID = gl.glCreateShader(GL3.GL_FRAGMENT_SHADER);
			gl.glShaderSource(fragmentShaderID, 1, new String[] { readAllLines("shaders/simulation.frag") }, null);
			gl.glCompileShader(fragmentShaderID);

			shaderProgramID = gl.glCreateProgram();
			gl.glAttachShader(shaderProgramID, vertexShaderID);
			gl.glAttachShader(shaderProgramID, fragmentShaderID);
			gl.glLinkProgram(shaderProgramID);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private static String readAllLines(String file) throws IOException {
		return Files.readAllLines(new File(file).toPath()).stream().reduce("", (accu, l) -> accu + l + System.lineSeparator());
	}

	/**
	 * Initialize the OpenGL VBO and the OpenCL VBO memory object
	 * 
	 * @param gl
	 *            The current GL object
	 */
	private void initVBOData(final GL3 gl) {
		initVBO(gl);
		simulation.initPositionBuffer(gl, vertexBufferObject);
	}

	/**
	 * Create the GL vertex buffer object (VBO) that stores the vertex positions.
	 * 
	 * @param gl
	 *            The GL context
	 */
	private void initVBO(final GL3 gl) {
		if (vertexBufferObject != 0) {
			gl.glDeleteBuffers(1, new int[] { vertexBufferObject }, 0);
			vertexBufferObject = 0;
		}
		if (vertexArrayObject != 0) {
			gl.glDeleteVertexArrays(1, new int[] { vertexArrayObject }, 0);
			vertexArrayObject = 0;
		}
		final int tempArray[] = new int[1];

		// Create the vertex array object
		gl.glGenVertexArrays(1, IntBuffer.wrap(tempArray));
		vertexArrayObject = tempArray[0];
		gl.glBindVertexArray(vertexArrayObject);

		// Create the vertex buffer object
		gl.glGenBuffers(1, IntBuffer.wrap(tempArray));
		vertexBufferObject = tempArray[0];

		// Initialize the vertex buffer object
		gl.glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObject);
		final int size = simulation.getNumberOfBodies() * Sizeof.cl_float4;
		gl.glBufferData(GL_ARRAY_BUFFER, size, null, GL_DYNAMIC_DRAW);

		// Initialize the attribute location of the input
		// vertices for the shader program
		final int location = gl.glGetAttribLocation(shaderProgramID, "inVertex");
		gl.glVertexAttribPointer(location, 4, GL3.GL_FLOAT, false, 0, 0);
		gl.glEnableVertexAttribArray(location);

	}

	/**
	 * Implementation of GLEventListener: Called when the given GLAutoDrawable is to be displayed.
	 */
	@Override
	public void display(final GLAutoDrawable drawable) {
		if (!initialized) {
			return;
		}
		final GL3 gl = (GL3) drawable.getGL();

		// run computation
		// Map OpenGL buffer object for writing from OpenCL
		gl.glFinish();
		simulation.step();

		gl.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Activate the shader program
		gl.glUseProgram(shaderProgramID);

		// Set the current projection matrix
		final int projectionMatrixLocation = gl.glGetUniformLocation(shaderProgramID, "projectionMatrix");
		gl.glUniformMatrix4fv(projectionMatrixLocation, 1, false, projectionMatrix, 0);

		// Set the current modelview matrix
		final int modelviewMatrixLocation = gl.glGetUniformLocation(shaderProgramID, "modelviewMatrix");
		gl.glUniformMatrix4fv(modelviewMatrixLocation, 1, false, modelviewMatrix, 0);

		final int screenSizeLocation = gl.glGetUniformLocation(shaderProgramID, "screenSize");
		gl.glUniform2f(screenSizeLocation, glComponent.getWidth(), glComponent.getHeight());

		final int spriteSizeLocation = gl.glGetUniformLocation(shaderProgramID, "spriteSize");
		gl.glUniform1f(spriteSizeLocation, 0.01f);

		// Render the VBO
		gl.glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObject);
		gl.glDrawArrays(GL_POINTS, 0, simulation.getNumberOfBodies());
	}

	/**
	 * Implementation of GLEventListener: Called then the GLAutoDrawable was reshaped
	 */
	@Override
	public void reshape(final GLAutoDrawable drawable, final int x, final int y, final int width, final int height) {
		setupView(drawable);
	}

	/**
	 * Set up a default view for the given GLAutoDrawable
	 * 
	 * @param drawable
	 *            The GLAutoDrawable to set the view for
	 */
	private void setupView(final GLAutoDrawable drawable) {
		final GL3 gl = (GL3) drawable.getGL();

		gl.glViewport(0, 0, drawable.getSurfaceWidth(), drawable.getSurfaceHeight());

		final float aspect = (float) drawable.getSurfaceWidth() / drawable.getSurfaceHeight();
		projectionMatrix = perspective(50, aspect, 0.1f, 100.0f);
	}

	@Override
	public void dispose(final GLAutoDrawable drawable) {
	}

	/**
	 * Calls System.exit() in a new Thread. It may not be called synchronously inside one of the JOGL callbacks.
	 * 
	 */
	private void runExit() {
		new Thread(new Runnable() {
			@Override
			public void run() {
				animator.stop();
				System.exit(0);
			}
		}).start();
	}

	// === Helper functions for matrix operations ==============================

	/**
	 * Helper method that creates a perspective matrix
	 * 
	 * @param fovy
	 *            The fov in y-direction, in degrees
	 * 
	 * @param aspect
	 *            The aspect ratio
	 * @param zNear
	 *            The near clipping plane
	 * @param zFar
	 *            The far clipping plane
	 * @return A perspective matrix
	 */
	private static float[] perspective(final float fovy, final float aspect, final float zNear, final float zFar) {
		final float radians = (float) Math.toRadians(fovy / 2);
		final float deltaZ = zFar - zNear;
		final float sine = (float) Math.sin(radians);
		if ((deltaZ == 0) || (sine == 0) || (aspect == 0)) {
			return identity();
		}
		final float cotangent = (float) Math.cos(radians) / sine;
		final float m[] = identity();
		m[0 * 4 + 0] = cotangent / aspect;
		m[1 * 4 + 1] = cotangent;
		m[2 * 4 + 2] = -(zFar + zNear) / deltaZ;
		m[2 * 4 + 3] = -1;
		m[3 * 4 + 2] = -2 * zNear * zFar / deltaZ;
		m[3 * 4 + 3] = 0;
		return m;
	}

	/**
	 * Creates an identity matrix
	 * 
	 * @return An identity matrix
	 */
	private static float[] identity() {
		final float m[] = new float[16];
		Arrays.fill(m, 0);
		m[0] = m[5] = m[10] = m[15] = 1.0f;
		return m;
	}

	/**
	 * Multiplies the given matrices and returns the result
	 * 
	 * @param m0
	 *            The first matrix
	 * @param m1
	 *            The second matrix
	 * @return The product m0*m1
	 */
	private static float[] multiply(final float m0[], final float m1[]) {
		final float m[] = new float[16];
		for (int x = 0; x < 4; x++) {
			for (int y = 0; y < 4; y++) {
				m[x * 4 + y] = m0[x * 4 + 0] * m1[y + 0] + m0[x * 4 + 1] * m1[y + 4] + m0[x * 4 + 2] * m1[y + 8] + m0[x * 4 + 3] * m1[y + 12];
			}
		}
		return m;
	}

	/**
	 * Creates a translation matrix
	 * 
	 * @param x
	 *            The x translation
	 * @param y
	 *            The y translation
	 * @param z
	 *            The z translation
	 * @return A translation matrix
	 */
	private static float[] translation(final float x, final float y, final float z) {
		final float m[] = identity();
		m[12] = x;
		m[13] = y;
		m[14] = z;
		return m;
	}

	/**
	 * Creates a matrix describing a rotation around the x-axis
	 * 
	 * @param angleDeg
	 *            The rotation angle, in degrees
	 * @return The rotation matrix
	 */
	private static float[] rotationX(final float angleDeg) {
		final float m[] = identity();
		final float angleRad = (float) Math.toRadians(angleDeg);
		final float ca = (float) Math.cos(angleRad);
		final float sa = (float) Math.sin(angleRad);
		m[5] = ca;
		m[6] = sa;
		m[9] = -sa;
		m[10] = ca;
		return m;
	}

	/**
	 * Creates a matrix describing a rotation around the y-axis
	 * 
	 * @param angleDeg
	 *            The rotation angle, in degrees
	 * @return The rotation matrix
	 */
	private static float[] rotationY(final float angleDeg) {
		final float m[] = identity();
		final float angleRad = (float) Math.toRadians(angleDeg);
		final float ca = (float) Math.cos(angleRad);
		final float sa = (float) Math.sin(angleRad);
		m[0] = ca;
		m[2] = -sa;
		m[8] = sa;
		m[10] = ca;
		return m;
	}

	public static void main(final String args[]) {
		final GLProfile profile = GLProfile.get(GLProfile.GL3);
		final GLCapabilities capabilities = new GLCapabilities(profile);
		SwingUtilities.invokeLater(new Runnable() {
			public void run() {
				new NBodyVisualizer(capabilities);
			}
		});
	}

}