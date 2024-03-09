import numpy as np
class ori():
    __slots__ = ('ea','q','aa')
    def __init__(self):
        self.ea = None
        self.q  = None
        self.aa = None

    def
    def q_from_ea(cls, ph1: float, phi: float, ph2: float) -> 'Quat':
        """
        CREDIT: from quat package from DefDAP
        NOTE: Below information reprodiced verbatim from quat

        Create a quat object from 3 Bunge euler angles.
        Parameters
        ----------
        ph1: First Euler angle, rotation around Z in radians.
        phi: Second Euler angle, rotation around new X in radians.
        ph2: Third Euler angle, rotation around new Z in radians.

        Returns
        -------
        defdap.quat.Quat
            Initialised Quat object.

        """
        # calculate quat coefficients
        quatCoef = np.array([
            np.cos(phi / 2.0) * np.cos((ph1 + ph2) / 2.0),
            -np.sin(phi / 2.0) * np.cos((ph1 - ph2) / 2.0),
            -np.sin(phi / 2.0) * np.sin((ph1 - ph2) / 2.0),
            -np.cos(phi / 2.0) * np.sin((ph1 + ph2) / 2.0)
        ], dtype=float)
        return quatCoef
    #-----------------------------------------------
    def fromAxisAngle(cls, axis: np.ndarray, angle: float) -> 'Quat':
        """
        CREDIT: from quat package from DefDAP
        NOTE: Below information reprodiced verbatim from quat

        Create a quat object from a rotation around an axis. This
        creates a quaternion to represent the passive rotation (-ve axis).

        Parameters
        ----------
        axis
            Axis that the rotation is applied around.
        angle
            Magnitude of rotation in radians.

        Returns
        -------
        defdap.quat.Quat
            Initialised Quat object.

        """
        # normalise the axis vector
        axis = np.array(axis)
        axis = axis / np.sqrt(np.dot(axis, axis))
        # calculate quat coefficients
        quatCoef = np.zeros(4, dtype=float)
        quatCoef[0] = np.cos(angle / 2)
        quatCoef[1:4] = -np.sin(angle / 2) * axis

        # call constructor
        return cls(quatCoef)
    #-----------------------------------------------
    def eulerAngles(self) -> np.ndarray:
        """
        CREDIT: from quat package from DefDAP
        NOTE: Below information reprodiced verbatim from quat

        Calculate the Euler angle representation for this rotation.

        Returns
        -------
        eulers : numpy.ndarray, shape 3
            Bunge euler angles (in radians).

        References
        ----------
            Melcher A. et al., 'Conversion of EBSD data by a quaternion
            based algorithm to be used for grain structure simulations',
            Technische Mechanik, 30(4)401 – 413

            Rowenhorst D. et al., 'Consistent representations of and
            conversions between 3D rotations',
            Model. Simul. Mater. Sci. Eng., 23(8)

        """
        eulers = np.empty(3, dtype=float)

        q = self.quatCoef
        q03 = q[0]**2 + q[3]**2
        q12 = q[1]**2 + q[2]**2
        chi = np.sqrt(q03 * q12)

        if chi == 0 and q12 == 0:
            eulers[0] = np.arctan2(-2 * q[0] * q[3], q[0]**2 - q[3]**2)
            eulers[1] = 0
            eulers[2] = 0

        elif chi == 0 and q03 == 0:
            eulers[0] = np.arctan2(2 * q[1] * q[2], q[1]**2 - q[2]**2)
            eulers[1] = np.pi
            eulers[2] = 0

        else:
            cosPh1 = (-q[0] * q[1] - q[2] * q[3]) / chi
            sinPh1 = (-q[0] * q[2] + q[1] * q[3]) / chi

            cosPhi = q[0]**2 + q[3]**2 - q[1]**2 - q[2]**2
            sinPhi = 2 * chi

            cosPh2 = (-q[0] * q[1] + q[2] * q[3]) / chi
            sinPh2 = (q[1] * q[3] + q[0] * q[2]) / chi

            eulers[0] = np.arctan2(sinPh1, cosPh1)
            eulers[1] = np.arctan2(sinPhi, cosPhi)
            eulers[2] = np.arctan2(sinPh2, cosPh2)

        if eulers[0] < 0:
            eulers[0] += 2 * np.pi
        if eulers[2] < 0:
            eulers[2] += 2 * np.pi

        return eulers
    #-----------------------------------------------
    def rotMatrix(self) -> np.ndarray:
        """
        CREDIT: from quat package from DefDAP
        NOTE: Below information reprodiced verbatim from quat

        Calculate the rotation matrix representation for this rotation.

        Returns
        -------
        rotMatrix : numpy.ndarray, shape (3, 3)
            Rotation matrix.

        References
        ----------
            Melcher A. et al., 'Conversion of EBSD data by a quaternion
            based algorithm to be used for grain structure simulations',
            Technische Mechanik, 30(4)401 – 413

            Rowenhorst D. et al., 'Consistent representations of and
            conversions between 3D rotations',
            Model. Simul. Mater. Sci. Eng., 23(8)

        """
        rotMatrix = np.empty((3, 3), dtype=float)

        q = self.quatCoef
        qbar = q[0]**2 - q[1]**2 - q[2]**2 - q[3]**2

        rotMatrix[0, 0] = qbar + 2 * q[1]**2
        rotMatrix[0, 1] = 2 * (q[1] * q[2] - q[0] * q[3])
        rotMatrix[0, 2] = 2 * (q[1] * q[3] + q[0] * q[2])

        rotMatrix[1, 0] = 2 * (q[1] * q[2] + q[0] * q[3])
        rotMatrix[1, 1] = qbar + 2 * q[2]**2
        rotMatrix[1, 2] = 2 * (q[2] * q[3] - q[0] * q[1])

        rotMatrix[2, 0] = 2 * (q[1] * q[3] - q[0] * q[2])
        rotMatrix[2, 1] = 2 * (q[2] * q[3] + q[0] * q[1])
        rotMatrix[2, 2] = qbar + 2 * q[3]**2

        return rotMatrix
    #-----------------------------------------------
