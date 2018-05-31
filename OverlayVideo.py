from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from numpy import *
from numpy.random import rand
import os
import subprocess
import sys
import time
import cStringIO
import tempfile
import pims
import pandas as pd
from PIL import Image
from itertools import product
from numpy.linalg import norm

class VideoOverlays:
    def __init__(self):
        pass
    @staticmethod
    def _clipVid(img, c=0.995):
        I = img.astype('float')
        I -= I.min()
        I /= I.max()
        h, Ibins = histogram(I.flatten(), 500)
        H = cumsum(h/sum(1.*h))
        Imin = Ibins[:-1][H<1.-c][-1] if any(H<1.-c) else 0.
        Imax = Ibins[1:][H>c][0] if any(H>c) else 1.
        I = I.clip(Imin, Imax)
        I -= I.min()
        I /= I.max()
        return I
    @staticmethod
    def _getColormap():
        cm = zeros((256, 3))
        cm[:, 0] = r_[linspace(0.5, 1.0, 30), ones(56), linspace(0., 1., 81)[::-1], zeros(89)][::-1]
        cm[:, 1] = r_[linspace(0.5, 1.0, 30), ones(56), linspace(0., 1., 81)[::-1], zeros(89)]
        cm[:, 2] = r_[zeros(31), linspace(0., 1., 66), ones(66), linspace(0., 1., 70)[::-1], zeros(23)]
        return cm[random.permutation(arange(256)), :]
    @staticmethod
    def _getParticleColors(trackData):
        pNumbers = array(list(set(int64(trackData.particle))))
        return uint8(255*array(particleColorMap[pNumbers % 256]))[..., :3]
    @staticmethod
    def _indicatorMask(xi, r):
        """Generate a tracking indicator shape as a boolian mask."""
        dist = norm(xi, axis=1)
        theta = arctan2(xi[:, 1], xi[:, 0])
        dtheta = 0.5
        out0 = (dist <= r+3)&(dist >= r)
        out1 = (theta > pi/4. - dtheta)&(theta < pi/4. + dtheta)
        out1 |= (theta > 3*pi/4. - dtheta)&(theta < 3*pi/4. + dtheta)
        out1 |= (theta > -pi/4. - dtheta)&(theta < -pi/4. + dtheta)
        out1 |= (theta > -3*pi/4. - dtheta)&(theta < -3*pi/4. + dtheta)
        return out0 & out1
    def _process(self, frames, trackData, outFileName, shape, r=13.):
        Nt, Ny, Nx = shape
        with tempfile.NamedTemporaryFile(mode='w+b', suffix='.mp4') as tmpfile:
            cmdstring = ('ffmpeg', '-y',
                         '-r', '10',
                         '-i', 'pipe:',
                         '-vcodec', 'libx264',
                         '-preset', 'slower',
                         '-profile:v', 'main',
                         '-crf', '30',
                         '-s', '{0}x{1}'.format(Nx, Ny),
                         '-pix_fmt', 'yuv420p',
                         tmpfile.name)
            proc = subprocess.Popen(cmdstring, stdin=subprocess.PIPE)
            xlocal = product(arange(-4*r, 4*r+1), arange(-4*r, 4*r+1))
            near = array([xn for xn in xlocal if norm(xn)<=r])
            particleColorMap = self._getColormap()
            particleColors = self._getParticleColors(trackData)
            tData = trackData.groupby('frame')
            timeSet = set(trackData.frame)
            for t in arange(Nt):
                frame = frames.get_frame(t)
                if isColor:
                    img = array(frame[:Ny, :Nx], 'uint16').max(axis=2)
                else:
                    img = array(frame[:Ny, :Nx], 'uint16')
                img = self._clipVid(img)
                img = uint8(255*img)
                I = array([img, img, img], 'uint8').transpose(1, 2, 0)
                if t in timeSet:
                    d = tData.get_group(t)
                    for p, x, y, rt in zip(d.particle, d.x, d.y, d.r):
                        indcs0 = around(array([x, y]) + near)
                        xi = indcs0 - array([x, y])
                        indcs = indcs0[self._indicatorMask(xi, rt)].astype(int)
                        indcs[indcs[:,0] >= Nx, 0] = Nx-1
                        indcs[indcs[:,1] >= Ny, 1] = Ny-1
                        indcs[indcs < 0] = 0
                        I[indcs[:, 1], indcs[:, 0], :] = particleColors[int(p)]
                f = cStringIO.StringIO()
                Image.fromarray(I).save(f, 'bmp')
                proc.stdin.write(f.getvalue())
                f.truncate(0)
            proc.stdin.close()
            proc.wait()
            tmpfile.seek(0)
            with open(outFileName, mode='w+b') as vfile:
                for line in tmpfile:
                    vfile.write(line)
    def makeOverlayVid(self, vidFileName, trackFileName):
        path, name = os.path.split(fileName)
        vname, ext = os.path.splitext(name)
        outFileName = os.path.join(path, vname + '-withOverlay' + ext)
        trackData = pd.read_csv(trackFileName)
        with pims.open(fileName) as frames:
            fshape = frames.frame_shape
            assert len(fshape) == 2 or len(fshape) == 3
            Nframes = len(frames)
            isColor = len(fshape) == 3 and Nframes > 1
            isImageJ_HyperStack = len(fshape) == 3 and Nframes == 1
            assert isImageJ_HyperStack == False
            Ny, Nx = fshape[0], fshape[1]
            Ny -= Ny % 2
            Nx -= Nx % 2
            Nt = Nframes
            self._process(
                frames,
                trackData,
                outFileName,
                (Nt, Ny, Nx),
                isColor)
        return outFileName
