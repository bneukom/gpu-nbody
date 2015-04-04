package ch.fhnw.woipv.nbody.opencl;

import static org.jocl.CL.*;

import java.io.Closeable;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.function.Predicate;
import java.util.stream.Collectors;

import org.jocl.cl_device_id;
import org.jocl.cl_platform_id;

public class CLPlatform implements Closeable {
	
	private cl_platform_id platformId;
	
	private CLPlatform(cl_platform_id id) {
		this.platformId = id;
	}
	
	/**
	 * Returns the number of available platforms.
	 * @return
	 */
	public static int getNumberOfPlatforms() {
		final int numPlatformsArray[] = new int[1];
		clGetPlatformIDs(0, null, numPlatformsArray);
		return numPlatformsArray[0];
	}
	
	/**
	 * Returns a {@link List} of all available opencl platforms.
	 * @return
	 */
	public static List<CLPlatform> getPlatforms() {
		final cl_platform_id platforms[] = new cl_platform_id[getNumberOfPlatforms()];
		clGetPlatformIDs(platforms.length, platforms, null);

		return Arrays.stream(platforms).map(x -> new CLPlatform(x)).collect(Collectors.toList());
	}
	
	/**
	 * Returns the first device matching the given filter or null if none exists.
	 * @param deviceType
	 * @param filter
	 * @return
	 */
	public CLDevice getDevice(long deviceType, Predicate<CLDevice> filter) {
		return getDevices(deviceType).stream().filter(filter).findFirst().orElse(null);
	}

	
	/**
	 * Returns all devices associated with this platform.
	 * @param deviceType
	 * @return
	 */
	public List<CLDevice> getDevices(long deviceType) {
		// Obtain the number of devices for the platform
		int numDevicesArray[] = new int[1];
		clGetDeviceIDs(platformId, deviceType, 0, null, numDevicesArray);
		int numDevices = numDevicesArray[0];

		// Obtain the all device IDs
		cl_device_id allDevices[] = new cl_device_id[numDevices];
		clGetDeviceIDs(platformId, deviceType, numDevices, allDevices, null);

		return Arrays.stream(allDevices).map(id ->  new CLDevice(id, this)).collect(Collectors.toList());
	}

	
	public cl_platform_id getPlatformId() {
		return platformId;
	}

	@Override
	public void close() throws IOException {
	}
	
	
}
