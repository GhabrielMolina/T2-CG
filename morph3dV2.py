"""
morph3d.py
Sistema de Morphing Geométrico 3D (T2-CG)

Este script implementa um sistema de morphing entre dois objetos 3D (.obj),
seguindo os requisitos típicos de trabalhos de Computação Gráfica.

Funcionalidades:
 - Leitura de arquivos OBJ (vértices e faces triangulares)
 - Normalização dos objetos (centralização e ajuste de escala)
 - Associação de faces por centroides (método sugerido no enunciado)
 - Interpolação linear (morphing) entre os dois objetos
 - Visualização em 3 janelas independentes (objeto 1, objeto 2 e morph)

Controles:
  m -> abrir/iniciar a janela do morph
  SPACE -> pausar / retomar a animação
  r -> resetar a animação (voltar ao início)
  ESC -> sair
  ← → ↑ ↓ -> rotacionar os objetos

Dependências:
 - PyOpenGL (OpenGL.GLUT, OpenGL.GLU, OpenGL.GL)
 - math
"""

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import sys
import math
import time

# === Configurações iniciais: altere aqui os nomes dos OBJ que deseja usar ===
OBJ_FILE_1 = "easy1.obj"
OBJ_FILE_2 = "easy3.obj"
# ===========================================================================


# ============================================================
# CLASSE Ponto
# Representa um vértice 3D com coordenadas (x, y, z)
# Inclui sobrecarga de operadores para facilitar cálculos vetoriais.
# ============================================================
class Ponto:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def as_tuple(self):
        """Retorna o ponto como uma tupla (x, y, z), usada no OpenGL."""
        return (self.x, self.y, self.z)

    def __add__(self, other):
        """Sobrecarga do operador '+'. Soma coordenada a coordenada."""
        return Ponto(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        """Sobrecarga do operador '-'. Subtrai coordenada a coordenada."""
        return Ponto(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, s):
        """Sobrecarga do operador '*'. Multiplica por um escalar."""
        return Ponto(self.x * s, self.y * s, self.z * s)
    __rmul__ = __mul__  # permite também s * Ponto

    def __repr__(self):
        """Retorno textual do ponto, útil para debug."""
        return f"Ponto({self.x}, {self.y}, {self.z})"


# ============================================================
# CLASSE Face
# Representa uma face triangular, armazenando 3 índices de vértices.
# ============================================================
class Face:
    """Face triangular representada por índices (i0,i1,i2) referenciando vértices."""
    def __init__(self, idx0, idx1, idx2):
        # OBJ usa índices baseados em 1, então armazenamos assim por padrão.
        self.idx = (int(idx0), int(idx1), int(idx2))

    def indices0(self):
        """
        Converte os índices para base-0 (Python usa listas baseadas em 0).
        Necessário para acessar corretamente as listas de vértices em Python.
        """
        return (self.idx[0]-1, self.idx[1]-1, self.idx[2]-1)


# ============================================================
# CLASSE Objeto3D
# Responsável por armazenar vértices e faces, ler arquivos OBJ,
# normalizar e calcular centroides.
# ============================================================
class Objeto3D:
    def __init__(self):
        self.vertices = []  # Lista de vértices originais (Ponto)
        self.faces = []     # Lista de faces (Face)
        self.transformed_vertices = []  # Vértices após normalização

    def LoadFile(self, filename):
        """
        Lê um arquivo OBJ simples e popula listas de vértices e faces.

        Linhas iniciadas com:
        - 'v' → definem vértices (x y z)
        - 'f' → definem faces (índices dos vértices)

        OBS: O formato OBJ pode usar:
        f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
        Aqui, extraímos apenas o índice do vértice (v), ignorando vt e vn.

        Também aplicamos triangulação por leque ("fan triangulation") para
        converter faces com mais de 3 vértices em triângulos.
        """
        self.vertices = []
        self.faces = []
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue  # ignora linhas vazias ou comentários

                parts = line.split()
                if parts[0] == 'v':
                    # Lê vértices: linhas do tipo "v x y z"
                    if len(parts) >= 4:
                        x, y, z = parts[1], parts[2], parts[3]
                        self.vertices.append(Ponto(float(x), float(y), float(z)))

                elif parts[0] == 'f':
                    # Lê faces: linhas do tipo "f v1 v2 v3" ou "f v1/vt1/vn1 ..."
                    verts = []
                    for tok in parts[1:]:
                        # Extrai apenas o índice antes da primeira '/'
                        if '/' in tok:
                            idx = tok.split('/')[0]
                        else:
                            idx = tok
                        if idx == '':
                            continue
                        verts.append(int(idx))

                    # Triangulação por leque:
                    # Se a face tiver mais de 3 vértices (ex: quadrado),
                    # dividimos em triângulos (v0,v1,v2), (v0,v2,v3), ...
                    if len(verts) < 3:
                        continue
                    for i in range(1, len(verts)-1):
                        self.faces.append(Face(verts[0], verts[i], verts[i+1]))

        # Copia os vértices originais para transformed_vertices (para normalização)
        self.transformed_vertices = [Ponto(v.x, v.y, v.z) for v in self.vertices]

    def compute_bbox(self):
        """Calcula a caixa delimitadora (bounding box) do objeto."""
        if not self.vertices:
            return None
        xs = [v.x for v in self.vertices]
        ys = [v.y for v in self.vertices]
        zs = [v.z for v in self.vertices]
        minp = Ponto(min(xs), min(ys), min(zs))
        maxp = Ponto(max(xs), max(ys), max(zs))
        return minp, maxp

    def center(self):
        """Retorna o centro do bounding box (usado para transladar o objeto à origem)."""
        bbox = self.compute_bbox()
        if bbox is None:
            return Ponto(0,0,0)
        minp, maxp = bbox
        return Ponto((minp.x+maxp.x)/2.0, (minp.y+maxp.y)/2.0, (minp.z+maxp.z)/2.0)

    def scale_factor(self):
        """
        Calcula o fator de escala para normalização.
        O inverso da maior dimensão do bounding box.
        """
        bbox = self.compute_bbox()
        if bbox is None:
            return 1.0
        minp, maxp = bbox
        sx = maxp.x - minp.x
        sy = maxp.y - minp.y
        sz = maxp.z - minp.z
        maxextent = max(sx, sy, sz)
        if maxextent == 0:
            return 1.0
        return 1.0 / maxextent

    def apply_transform(self, translate=Ponto(0,0,0), scale=1.0):
        """
        Aplica transformação de normalização:
        - Translada o objeto para a origem
        - Aplica o fator de escala (ajuste de tamanho)
        """
        self.transformed_vertices = []
        for v in self.vertices:
            tv = Ponto(v.x - translate.x, v.y - translate.y, v.z - translate.z)
            tv = tv * scale
            self.transformed_vertices.append(tv)

    def get_face_centroid(self, face_idx):
        """
        Calcula o centroide (média das coordenadas) dos três vértices
        que formam uma face triangular.
        """
        f = self.faces[face_idx]
        i0,i1,i2 = f.indices0()
        v0 = self.transformed_vertices[i0]
        v1 = self.transformed_vertices[i1]
        v2 = self.transformed_vertices[i2]
        cx = (v0.x + v1.x + v2.x)/3.0
        cy = (v0.y + v1.y + v2.y)/3.0
        cz = (v0.z + v1.z + v2.z)/3.0
        return Ponto(cx, cy, cz)


# ============================================================
# CLASSE MorphSystem
# Controla a lógica central do morph:
# - Leitura dos objetos
# - Normalização e alinhamento
# - Associação de faces por centroides
# - Interpolação dos vértices durante a animação
# ============================================================
class MorphSystem:
    def __init__(self):
        # Objetos 3D de entrada
        self.obj1 = Objeto3D()
        self.obj2 = Objeto3D()
        # Dicionário de mapeamento: face do obj1 → face correspondente do obj2
        self.mapping = {}
        # Parâmetros da animação
        self.num_frames = 100   # número de frames totais
        self.t = 0.0            # parâmetro de interpolação (0 a 1)
        self.frame = 0          # frame atual
        self.playing = False    # indica se a animação está rodando
        # Referências às janelas OpenGL
        self.window1 = None
        self.window2 = None
        self.window3 = None
        # Ângulos de rotação usados para interação com o teclado
        self.rotY = 0.0
        self.rotX = 0.0

    def load_objects(self, file1, file2):
        """Carrega os dois arquivos OBJ e inicia normalização e associação."""
        print("Loading OBJ 1:", file1)
        self.obj1.LoadFile(file1)
        print("Vertices:", len(self.obj1.vertices), "Faces:", len(self.obj1.faces))

        print("Loading OBJ 2:", file2)
        self.obj2.LoadFile(file2)
        print("Vertices:", len(self.obj2.vertices), "Faces:", len(self.obj2.faces))

        # Normaliza ambos os objetos e alinha seus centros
        self.normalize_and_align()

        # Cria o mapeamento de faces (face a face)
        self.associate_faces_by_centroid()

    def normalize_and_align(self):
        """
        Normaliza os dois objetos:
        - Centraliza ambos na origem (translação)
        - Escala ambos para tamanho unitário (com base no bounding box)
        Assim, ambos terão tamanhos comparáveis para o morph.
        """
        sf1 = self.obj1.scale_factor()
        sf2 = self.obj2.scale_factor()
        c1 = self.obj1.center()
        c2 = self.obj2.center()
        self.obj1.apply_transform(translate=c1, scale=sf1)
        self.obj2.apply_transform(translate=c2, scale=sf2)

    def associate_faces_by_centroid(self):
        """
        Implementa o requisito 3 do enunciado: Associação de faces.
        Para cada face F1 do objeto 1, encontra a face F2 do objeto 2 cujo
        centroide está mais próximo (menor distância euclidiana).
        O resultado é armazenado em self.mapping, permitindo associação N→1.
        """
        n1 = len(self.obj1.faces)
        n2 = len(self.obj2.faces)
        if n1 == 0 or n2 == 0:
            print("Warning: um dos objetos não possui faces.")
            return

        # Pré-calcula centroides do objeto 2
        centroids2 = [self.obj2.get_face_centroid(i) for i in range(n2)]

        # Para cada face do objeto 1, encontra a mais próxima no objeto 2
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
            # Associa face i do obj1 à face best_j do obj2 (muitos-para-um)
            self.mapping[i] = best_j

        print(f"Associadas {n1} faces de OBJ1 às {n2} faces de OBJ2 (many-to-one permitido).")

    def reset_animation(self):
        """Reseta o estado da animação."""
        self.frame = 0
        self.t = 0.0
        self.playing = False

    def step_animation(self):
        """Avança um frame na animação se estiver rodando."""
        if not self.playing:
            return
        self.frame += 1
        if self.frame > self.num_frames:
            self.frame = self.num_frames
            self.playing = False
        self.t = float(self.frame) / float(self.num_frames)

    def get_interpolated_triangles(self):
        """
        Calcula a geometria interpolada (morph) no frame atual.
        Para cada par de faces associadas (F1,F2), interpola seus vértices.
        Fórmula da interpolação linear (LERP):
            V(t) = V1 + (V2 - V1) * t
        """
        tris = []
        t = self.t
        for f_idx1, f_idx2 in self.mapping.items():
            f1 = self.obj1.faces[f_idx1]
            f2 = self.obj2.faces[f_idx2]
            i0_1, i1_1, i2_1 = f1.indices0()
            i0_2, i1_2, i2_2 = f2.indices0()
            v1_0 = self.obj1.transformed_vertices[i0_1]
            v1_1 = self.obj1.transformed_vertices[i1_1]
            v1_2 = self.obj1.transformed_vertices[i2_1]
            v2_0 = self.obj2.transformed_vertices[i0_2]
            v2_1 = self.obj2.transformed_vertices[i1_2]
            v2_2 = self.obj2.transformed_vertices[i2_2]

            # Interpolação ponto a ponto (LERP)
            iv0 = Ponto(v1_0.x + (v2_0.x - v1_0.x)*t,
                        v1_0.y + (v2_0.y - v1_0.y)*t,
                        v1_0.z + (v2_0.z - v1_0.z)*t)
            iv1 = Ponto(v1_1.x + (v2_1.x - v1_1.x)*t,
                        v1_1.y + (v2_1.y - v1_1.y)*t,
                        v1_1.z + (v2_1.z - v1_1.z)*t)
            iv2 = Ponto(v1_2.x + (v2_2.x - v1_2.x)*t,
                        v1_2.y + (v2_2.y - v1_2.y)*t,
                        v1_2.z + (v2_2.z - v1_2.z)*t)
            tris.append((iv0, iv1, iv2))
        return tris


# ============================================================
# SEÇÃO OPENGL / GLUT
# Define janelas, funções de renderização e controle de animação.
# ============================================================
morph = MorphSystem()  # instância global usada pelos callbacks

# Tamanhos iniciais das janelas
window1_size = [400, 400]
window2_size = [400, 400]
window3_size = [600, 600]


def setup_projection(width, height):
    """Configura a projeção em perspectiva para manter o aspecto da janela."""
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    aspect = float(width)/float(height) if height!=0 else 1.0
    gluPerspective(45.0, aspect, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def draw_axes():
    """Desenha eixos X (vermelho), Y (verde) e Z (azul) para referência."""
    glBegin(GL_LINES)
    glColor3f(1.0, 0.0, 0.0); glVertex3f(0,0,0); glVertex3f(0.5,0,0)
    glColor3f(0.0, 1.0, 0.0); glVertex3f(0,0,0); glVertex3f(0,0.5,0)
    glColor3f(0.0, 0.0, 1.0); glVertex3f(0,0,0); glVertex3f(0,0,0.5)
    glEnd()


def display_obj(obj: Objeto3D, win_w, win_h):
    """
    Função de desenho usada nas janelas 1 e 2.
    Desenha o objeto fornecido em modo wireframe (somente linhas).
    """
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    # Configura a câmera
    gluLookAt(0.0, 0.0, 2.5,   0.0, 0.0, 0.0,   0.0, 1.0, 0.0)
    glRotatef(morph.rotX, 1.0, 0.0, 0.0)
    glRotatef(morph.rotY, 0.0, 1.0, 0.0)
    draw_axes()

    # Desenha as faces como linhas
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


# === Funções de exibição de cada janela ===
def display_win1():
    """Renderiza a Janela 1 (Objeto 1)."""
    glutSetWindow(morph.window1)
    glViewport(0,0,window1_size[0], window1_size[1])
    display_obj(morph.obj1, window1_size[0], window1_size[1])


def display_win2():
    """Renderiza a Janela 2 (Objeto 2)."""
    glutSetWindow(morph.window2)
    glViewport(0,0,window2_size[0], window2_size[1])
    display_obj(morph.obj2, window2_size[0], window2_size[1])


def display_win3():
    """
    Renderiza a Janela 3 (Animação do Morphing).
    1. Obtém os triângulos interpolados.
    2. Define cor baseada em t.
    3. Desenha triângulos preenchidos + contorno wireframe.
    """
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

    tris = morph.get_interpolated_triangles()

    # Preenchimento (GL_FILL)
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    glBegin(GL_TRIANGLES)
    glColor3f(0.8*(1.0-morph.t)+0.2*morph.t, 0.2, 0.8*morph.t + 0.2*(1.0-morph.t))
    for (v0,v1,v2) in tris:
        glVertex3f(v0.x, v0.y, v0.z)
        glVertex3f(v1.x, v1.y, v1.z)
        glVertex3f(v2.x, v2.y, v2.z)
    glEnd()

    # Sobreposição do contorno em modo wireframe
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


# === Funções de redimensionamento (reshape) ===
def reshape_win1(w,h):
    """Mantém proporção ao redimensionar a Janela 1."""
    window1_size[0] = max(1,w); window1_size[1] = max(1,h)
    glutSetWindow(morph.window1)
    glViewport(0,0,w,h)
    setup_projection(w,h)

def reshape_win2(w,h):
    """Mantém proporção ao redimensionar a Janela 2."""
    window2_size[0] = max(1,w); window2_size[1] = max(1,h)
    glutSetWindow(morph.window2)
    glViewport(0,0,w,h)
    setup_projection(w,h)

def reshape_win3(w,h):
    """Mantém proporção ao redimensionar a Janela 3."""
    window3_size[0] = max(1,w); window3_size[1] = max(1,h)
    glutSetWindow(morph.window3)
    glViewport(0,0,w,h)
    setup_projection(w,h)


# === Função Idle ===
def idle_func():
    """
    Função chamada continuamente pelo GLUT.
    É o "motor" da animação: avança frames, atualiza self.t e solicita redesenho.
    """
    if morph.playing and morph.window3 is not None:
        morph.step_animation()
        glutSetWindow(morph.window3)
        glutPostRedisplay()
    time.sleep(0.01)  # pequena pausa para evitar uso total da CPU


# === Controles de teclado ===
def keyboard_common(key, x, y):
    """
    Teclas:
      ESC -> sair
      m -> abrir e iniciar janela de morph
      espaço -> pausar/retomar animação
      r -> resetar morph
    """
    k = key.decode('utf-8') if isinstance(key, bytes) else key
    if k == '\x1b':  # ESC
        print("Exiting.")
        sys.exit(0)
    if k == 'm':
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
    """Permite rotacionar os objetos com as setas do teclado."""
    if key == GLUT_KEY_LEFT:
        morph.rotY -= 5.0
    elif key == GLUT_KEY_RIGHT:
        morph.rotY += 5.0
    elif key == GLUT_KEY_UP:
        morph.rotX -= 5.0
    elif key == GLUT_KEY_DOWN:
        morph.rotX += 5.0
    glutPostRedisplay()


# === Criação das janelas ===
def create_window1():
    """Cria a Janela 1 (objeto 1)."""
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
    """Cria a Janela 2 (objeto 2)."""
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
    """Cria a Janela 3 (animação do morph)."""
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


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================
def main():
    """
    Fluxo principal:
    1. Carrega os dois objetos (.obj)
    2. Normaliza e associa as faces
    3. Inicializa o GLUT
    4. Cria as janelas 1 e 2
    5. Define idle_func para gerenciar a animação
    6. Inicia o loop principal do GLUT
    """
    if len(sys.argv) >= 3:
        file1 = sys.argv[1]
        file2 = sys.argv[2]
    else:
        file1 = OBJ_FILE_1
        file2 = OBJ_FILE_2

    morph.load_objects(file1, file2)
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
    create_window1()
    create_window2()
    glutIdleFunc(idle_func)
    print("Controles: m=iniciar morph | SPACE=pause | r=reset | ESC=sair")
    glutMainLoop()


if __name__ == "__main__":
    main()
