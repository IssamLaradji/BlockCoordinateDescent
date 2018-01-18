import numpy as np 
import scipy.linalg as sp

def step(x, d, block, proj):
  xc = x.copy()
  xc[block] = proj(x[block] + d)
  return xc


def perform_line_search(x_old, grad, block, 
                        F, D, 
                        alpha0=1.0, 
                        proj=None):
    x = x_old.copy()
    g = grad
    

    proj = proj or (lambda x: x)

    
    Z = lambda x, phi, alpha: F(step(x, D(alpha), block, proj)) - F(x) - alpha * phi

    t = 1
    eps = 1e-10
    alpha = alpha0
    phi = np.dot(g, D(alpha))

    while F(step(x, D(alpha), block, proj)) > (F(x) + eps * alpha * phi):
        
        alphaTmp = alpha
        if t == 1:
          # Quadratic interpolation
          z = Z(x, phi, alpha)
          alpha =  - (phi*alpha**2) / (2 * z)

          # Keep track
          zOld = z
          alphaOld = alphaTmp

        elif t > 1:
          # Cubic interpolation
          c = 1. / ((alpha - alphaOld) * (alphaOld**2) * alpha**2 )
          z = Z(x, phi, alpha)

          a = c * ((alphaOld**2) * z - (alpha**2) * zOld)
          b = c * ((-alphaOld**3) * z + (alpha**3) * zOld)
         
          alpha = (-b + np.sqrt(b**2 - 3 * a * phi)) / (3 * a)

          # Keep track
          zOld = z
          alphaOld = alphaTmp

        t += 1

    return alpha

