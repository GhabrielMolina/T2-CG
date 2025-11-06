"""
morph3d.py
Sistema de Morphing Geométrico 3D (T2-CG)
Implementa:
 - leitura simples de .obj (v, f)
 - normalização por bounding-box e alinhamento de centros
 - associação de faces por centroides (many-to-one permitido)
 - 3 janelas GLUT: janela1(obj1), janela2(obj2), janela3(morph animado)
 - interpolação linear de vértices entre faces associadas
Controls:
  m -> abrir Janela 3 / iniciar morph
  SPACE -> pausar / retomar
  r -> reset
  ESC -> sair
Dependencies: PyOpenGL (OpenGL.GLUT, OpenGL.GLU, OpenGL.GL), math
"""
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import sys
import math
import time

# === Configurações iniciais: altere aqui os nomes dos OBJ que deseja usar ===
OBJ_FILE_1 = "easy1.obj"
OBJ_FILE_2 = "easy2.obj"  # troque pelo arquivo desejado
# ===========================================================================

# ----------------------------
# Classes utilitárias
# ----------------------------
class Ponto:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
    def as_tuple(self):
        return (self.x, self.y, self.z)
    def __add__(self, other):
        return Ponto(self.x + other.x, self.y + other.y, self.z + other.z)
    def __sub__(self, other):
        return Ponto(self.x - other.x, self.y - other.y, self.z - other.z)
    def __mul__(self, s):
        return Ponto(self.x * s, self.y * s, self.z * s)
    __rmul__ = __mul__
    def __repr__(self):
        return f"Ponto({self.x}, {self.y}, {self.z})"

class Face:
    """Face triangular representada por índices (i0,i1,i2) referenciando vértices"""
    def __init__(self, idx0, idx1, idx2):
        self.idx = (int(idx0), int(idx1), int(idx2))  # 1-based indices from OBJ
    def indices0(self):
        # retorna 0-based indices
        return (self.idx[0]-1, self.idx[1]-1, self.idx[2]-1)

class Objeto3D:
    def __init__(self):
        self.vertices = []  # lista de Ponto
        self.faces = []     # lista de Face (triangulares)
        # after normalization store transformed vertices for drawing
        self.transformed_vertices = []

    def LoadFile(self, filename):
        """
        Lê um .obj simples:
         - linhas começando com 'v' contêm vertices (x y z)
         - linhas começando com 'f' contêm faces (indices). Trata formatos v, v/vt/vn, v//vn.
         - triangula faces com >3 vértices usando fan triangulation.
        """
        self.vertices = []
        self.faces = []
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if parts[0] == 'v':
                    # Pegar os três primeiros float depois de 'v'
                    if len(parts) >= 4:
                        x, y, z = parts[1], parts[2], parts[3]
                        self.vertices.append(Ponto(float(x), float(y), float(z)))
                elif parts[0] == 'f':
                    # cada token pode ser: v, v/vt, v//vn, v/vt/vn
                    verts = []
                    for tok in parts[1:]:
                        # extrair índice de vértice (antes da primeira '/')
                        if '/' in tok:
                            idx = tok.split('/')[0]
                        else:
                            idx = tok
                        if idx == '':
                            continue
                        verts.append(int(idx))
                    # triangular: fan triangulation
                    if len(verts) < 3:
                        continue
                    for i in range(1, len(verts)-1):
                        self.faces.append(Face(verts[0], verts[i], verts[i+1]))
        # inicializar transformed_vertices
        self.transformed_vertices = [Ponto(v.x, v.y, v.z) for v in self.vertices]

    def compute_bbox(self):
        if not self.vertices:
            return None
        xs = [v.x for v in self.vertices]
        ys = [v.y for v in self.vertices]
        zs = [v.z for v in self.vertices]
        minp = Ponto(min(xs), min(ys), min(zs))
        maxp = Ponto(max(xs), max(ys), max(zs))
        return minp, maxp

    def center(self):
        # centro do bounding box
        bbox = self.compute_bbox()
        if bbox is None:
            return Ponto(0,0,0)
        minp, maxp = bbox
        return Ponto((minp.x+maxp.x)/2.0, (minp.y+maxp.y)/2.0, (minp.z+maxp.z)/2.0)

    def scale_factor(self):
        bbox = self.compute_bbox()
        if bbox is None:
            return 1.0
        minp, maxp = bbox
        sx = maxp.x - minp.x
        sy = maxp.y - minp.y
        sz = maxp.z - minp.z
        # use maximum extent
        maxextent = max(sx, sy, sz)
        if maxextent == 0:
            return 1.0
        return 1.0 / maxextent

    def apply_transform(self, translate=Ponto(0,0,0), scale=1.0):
        self.transformed_vertices = []
        for v in self.vertices:
            tv = Ponto(v.x - translate.x, v.y - translate.y, v.z - translate.z)
            tv = tv * scale
            self.transformed_vertices.append(tv)

    def get_face_centroid(self, face_idx):
        f = self.faces[face_idx]
        i0,i1,i2 = f.indices0()
        v0 = self.transformed_vertices[i0]
        v1 = self.transformed_vertices[i1]
        v2 = self.transformed_vertices[i2]
        cx = (v0.x + v1.x + v2.x)/3.0
        cy = (v0.y + v1.y + v2.y)/3.0
        cz = (v0.z + v1.z + v2.z)/3.0
        return Ponto(cx, cy, cz)

# ----------------------------
# Morphing manager / lógica
# ----------------------------
class MorphSystem:
    def __init__(self):
        self.obj1 = Objeto3D()
        self.obj2 = Objeto3D()
        self.mapping = {}  # map face index in obj1 -> face index in obj2 (many-to-one)
        # animation params
        self.num_frames = 100
        self.t = 0.0
        self.frame = 0
        self.playing = False
        self.window1 = None
        self.window2 = None
        self.window3 = None
        # viewing angles
        self.rotY = 0.0
        self.rotX = 0.0

    def load_objects(self, file1, file2):
        print("Loading OBJ 1:", file1)
        self.obj1.LoadFile(file1)
        print("Vertices:", len(self.obj1.vertices), "Faces:", len(self.obj1.faces))
        print("Loading OBJ 2:", file2)
        self.obj2.LoadFile(file2)
        print("Vertices:", len(self.obj2.vertices), "Faces:", len(self.obj2.faces))

        # normalize and align centers
        self.normalize_and_align()

        # compute mapping
        self.associate_faces_by_centroid()

    def normalize_and_align(self):
        # normalize by bounding box scale and align centers to origin
        # We'll scale both objects so they fit in same unit scale (based on each own bbox),
        # then optionally scale them by smallest or largest? We'll scale so both have same final extents:
        sf1 = self.obj1.scale_factor()
        sf2 = self.obj2.scale_factor()
        # To keep comparable sizes: scale both by their own scale factors (so each max extent -> 1.0),
        # then we can optionally rescale both by 1.0 (they're both normalized).
        c1 = self.obj1.center()
        c2 = self.obj2.center()
        self.obj1.apply_transform(translate=c1, scale=sf1)
        self.obj2.apply_transform(translate=c2, scale=sf2)
        # After this, both objects occupy roughly the unit-box extents and are centered at origin.

    def associate_faces_by_centroid(self):
        n1 = len(self.obj1.faces)
        n2 = len(self.obj2.faces)
        if n1 == 0 or n2 == 0:
            print("Warning: one of the objects has zero faces.")
            return
        # precompute centroids for obj2
        centroids2 = [self.obj2.get_face_centroid(i) for i in range(n2)]
        # For each face in obj1, find nearest face in obj2
        self.mapping = {}
        for i in range(n1):
            c1 = self.obj1.get_face_centroid(i)
            best_j = 0
            best_dist = float('inf')
            for j in range(n2):
                c2 = centroids2[j]
                dx = c1.x - c2.x
                dy = c1.y - c2.y
                dz = c1.z - c2.z
                d2 = dx*dx + dy*dy + dz*dz
                if d2 < best_dist:
                    best_dist = d2
                    best_j = j
            self.mapping[i] = best_j
        print(f"Associated {n1} faces of OBJ1 to {n2} faces of OBJ2 (many-to-one allowed).")

    def reset_animation(self):
        self.frame = 0
        self.t = 0.0
        self.playing = False

    def step_animation(self):
        if not self.playing:
            return
        self.frame += 1
        if self.frame > self.num_frames:
            self.frame = self.num_frames
            self.playing = False
        self.t = float(self.frame) / float(self.num_frames)

    def get_interpolated_triangles(self):
        """
        Retorna uma lista de triângulos interpolados (cada triângulo = tuple de 3 Ponto),
        correspondendo ao conjunto de faces do obj1 mapeadas para faces do obj2, usando t.
        """
        tris = []
        t = self.t
        for f_idx1, f_idx2 in self.mapping.items():
            f1 = self.obj1.faces[f_idx1]
            f2 = self.obj2.faces[f_idx2]
            i0_1, i1_1, i2_1 = f1.indices0()
            i0_2, i1_2, i2_2 = f2.indices0()
            # obter vértices transformados correspondentes
            v1_0 = self.obj1.transformed_vertices[i0_1]
            v1_1 = self.obj1.transformed_vertices[i1_1]
            v1_2 = self.obj1.transformed_vertices[i2_1]
            v2_0 = self.obj2.transformed_vertices[i0_2]
            v2_1 = self.obj2.transformed_vertices[i1_2]
            v2_2 = self.obj2.transformed_vertices[i2_2]
            # interpolação linear por vértice
            # iv0 = Ponto(v1_0.x + (v2_0.x - v1_0.x)*t,
            #             v1_0.y + (v2_0.y - v1_0.y)*t,
            #             v1_0.z + (v2_0.z - v1_0.z)*t)
            # iv1 = Ponto(v1_1.x + (v2_1.x - v1_1.x)*t,
            #             v1_1.y + (v2_1.y - v1_1.y)*t,
            #             v1_1.z + (v2_1.z - v1_1.z)*t)
            # iv2 = Ponto(v1_2.x + (v2_2.x - v1_2.z if False else v2_2.x - v1_2.x)*t,
            #             v1_2.y + (v2_2.y - v1_2.y)*t,
            #             v1_2.z + (v2_2.z - v1_2.z)*t)
            # Note: above had a small typo risk; ensure correct formula consistently:
            # We'll recompute more defensively:
            iv0 = Ponto(v1_0.x + (v2_0.x - v1_0.x)*t, v1_0.y + (v2_0.y - v1_0.y)*t, v1_0.z + (v2_0.z - v1_0.z)*t)
            iv1 = Ponto(v1_1.x + (v2_1.x - v1_1.x)*t, v1_1.y + (v2_1.y - v1_1.y)*t, v1_1.z + (v2_1.z - v1_1.z)*t)
            iv2 = Ponto(v1_2.x + (v2_2.x - v1_2.x)*t, v1_2.y + (v2_2.y - v1_2.y)*t, v1_2.z + (v2_2.z - v1_2.z)*t)
            tris.append((iv0, iv1, iv2))
        return tris

# ----------------------------
# OpenGL / GLUT callbacks and windows
# ----------------------------
morph = MorphSystem()

# keep track of window sizes and aspect corrections
window1_size = [400, 400]
window2_size = [400, 400]
window3_size = [600, 600]

def setup_projection(width, height):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    aspect = float(width)/float(height) if height!=0 else 1.0
    gluPerspective(45.0, aspect, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def draw_axes():
    glBegin(GL_LINES)
    # x red
    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(0,0,0); glVertex3f(0.5,0,0)
    # y green
    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(0,0,0); glVertex3f(0,0.5,0)
    # z blue
    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(0,0,0); glVertex3f(0,0,0.5)
    glEnd()

def display_obj(obj: Objeto3D, win_w, win_h):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    # camera
    gluLookAt(0.0, 0.0, 2.5,   0.0, 0.0, 0.0,   0.0, 1.0, 0.0)
    glRotatef(morph.rotX, 1.0, 0.0, 0.0)
    glRotatef(morph.rotY, 0.0, 1.0, 0.0)
    # draw axes
    draw_axes()
    # draw triangles
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    glBegin(GL_TRIANGLES)
    glColor3f(1.0, 1.0, 1.0)
    for face in obj.faces:
        i0,i1,i2 = face.indices0()
        v0 = obj.transformed_vertices[i0]
        v1 = obj.transformed_vertices[i1]
        v2 = obj.transformed_vertices[i2]
        glVertex3f(v0.x, v0.y, v0.z)
        glVertex3f(v1.x, v1.y, v1.z)
        glVertex3f(v2.x, v2.y, v2.z)
    glEnd()
    glutSwapBuffers()

# GLUT callbacks per-window
def display_win1():
    glutSetWindow(morph.window1)
    glViewport(0,0,window1_size[0], window1_size[1])
    display_obj(morph.obj1, window1_size[0], window1_size[1])

def display_win2():
    glutSetWindow(morph.window2)
    glViewport(0,0,window2_size[0], window2_size[1])
    display_obj(morph.obj2, window2_size[0], window2_size[1])

def display_win3():
    if morph.window3 is None:
        return
    glutSetWindow(morph.window3)
    glViewport(0,0,window3_size[0], window3_size[1])
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluLookAt(0.0, 0.0, 2.5,  0.0,0.0,0.0,  0.0,1.0,0.0)
    glRotatef(morph.rotX, 1.0, 0.0, 0.0)
    glRotatef(morph.rotY, 0.0, 1.0, 0.0)
    draw_axes()
    # draw interpolated triangles
    tris = morph.get_interpolated_triangles()
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    glBegin(GL_TRIANGLES)
    # color gradient based on t
    glColor3f(0.8*(1.0-morph.t)+0.2*morph.t, 0.2, 0.8*morph.t + 0.2*(1.0-morph.t))
    for (v0,v1,v2) in tris:
        glVertex3f(v0.x, v0.y, v0.z)
        glVertex3f(v1.x, v1.y, v1.z)
        glVertex3f(v2.x, v2.y, v2.z)
    glEnd()
    # wireframe overlay
    glEnable(GL_POLYGON_OFFSET_FILL)
    glPolygonOffset(1.0, 1.0)
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    glColor3f(0.0,0.0,0.0)
    glBegin(GL_TRIANGLES)
    for (v0,v1,v2) in tris:
        glVertex3f(v0.x, v0.y, v0.z)
        glVertex3f(v1.x, v1.y, v1.z)
        glVertex3f(v2.x, v2.y, v2.z)
    glEnd()
    glDisable(GL_POLYGON_OFFSET_FILL)
    glutSwapBuffers()

def reshape_win1(w,h):
    window1_size[0] = max(1,w); window1_size[1] = max(1,h)
    glutSetWindow(morph.window1)
    glViewport(0,0,w,h)
    setup_projection(w,h)

def reshape_win2(w,h):
    window2_size[0] = max(1,w); window2_size[1] = max(1,h)
    glutSetWindow(morph.window2)
    glViewport(0,0,w,h)
    setup_projection(w,h)

def reshape_win3(w,h):
    window3_size[0] = max(1,w); window3_size[1] = max(1,h)
    glutSetWindow(morph.window3)
    glViewport(0,0,w,h)
    setup_projection(w,h)

# idle function used by animation
def idle_func():
    # drive animation steps if playing and window3 exists
    if morph.playing and morph.window3 is not None:
        morph.step_animation()
        # request redisplay of the third window
        glutSetWindow(morph.window3)
        glutPostRedisplay()
    time.sleep(0.01)  # small sleep to avoid 100% cpu

def keyboard_common(key, x, y):
    k = key.decode('utf-8') if isinstance(key, bytes) else key
    if k == '\x1b':  # ESC
        print("Exiting.")
        sys.exit(0)
    if k == 'm':
        # open window3 if not open, and start morph
        if morph.window3 is None:
            create_window3()
        morph.playing = True
        morph.frame = 0
        morph.t = 0.0
    elif k == ' ':
        morph.playing = not morph.playing
    elif k == 'r':
        morph.reset_animation()
        if morph.window3 is not None:
            glutPostRedisplay()
    glutPostRedisplay()

def special_common(key, x, y):
    # arrow keys to rotate
    if key == GLUT_KEY_LEFT:
        morph.rotY -= 5.0
    elif key == GLUT_KEY_RIGHT:
        morph.rotY += 5.0
    elif key == GLUT_KEY_UP:
        morph.rotX -= 5.0
    elif key == GLUT_KEY_DOWN:
        morph.rotX += 5.0
    glutPostRedisplay()

# Window creation
def create_window1():
    glutInitWindowSize(window1_size[0], window1_size[1])
    glutInitWindowPosition(100, 100)
    win = glutCreateWindow(b"OBJ 1")
    morph.window1 = win
    glEnable(GL_DEPTH_TEST)
    setup_projection(window1_size[0], window1_size[1])
    glutDisplayFunc(display_win1)
    glutReshapeFunc(reshape_win1)
    glutKeyboardFunc(keyboard_common)
    glutSpecialFunc(special_common)
    return win

def create_window2():
    glutInitWindowSize(window2_size[0], window2_size[1])
    glutInitWindowPosition(510, 100)
    win = glutCreateWindow(b"OBJ 2")
    morph.window2 = win
    glEnable(GL_DEPTH_TEST)
    setup_projection(window2_size[0], window2_size[1])
    glutDisplayFunc(display_win2)
    glutReshapeFunc(reshape_win2)
    glutKeyboardFunc(keyboard_common)
    glutSpecialFunc(special_common)
    return win

def create_window3():
    glutInitWindowSize(window3_size[0], window3_size[1])
    glutInitWindowPosition(200, 520)
    win = glutCreateWindow(b"Morphing Animation")
    morph.window3 = win
    glEnable(GL_DEPTH_TEST)
    setup_projection(window3_size[0], window3_size[1])
    glutDisplayFunc(display_win3)
    glutReshapeFunc(reshape_win3)
    glutKeyboardFunc(keyboard_common)
    glutSpecialFunc(special_common)
    return win

# ----------------------------
# Main
# ----------------------------
def main():
    if len(sys.argv) >= 3:
        file1 = sys.argv[1]
        file2 = sys.argv[2]
    else:
        file1 = OBJ_FILE_1
        file2 = OBJ_FILE_2
    # load objects and prepare mapping
    morph.load_objects(file1, file2)

    # initialize GLUT
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
    # Create windows 1 and 2 initially 400x400 each, side by side
    create_window1()
    create_window2()
    # register a global idle
    glutIdleFunc(idle_func)
    # start main loop
    print("Controls: m=open/start morph | SPACE=pause | r=reset | ESC=exit")
    glutMainLoop()

if __name__ == "__main__":
    main()
