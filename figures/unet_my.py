from pathlib import Path


TEX_TEMPLATE = r"""
\documentclass[border=10pt, tikz]{standalone}
\usetikzlibrary{arrows.meta}

% Draw a group of thin vertical slices
% #1=name, #2=x, #3=y, #4=number of slices, #5=height, #6=fill color
\newcommand{\slicegroup}[6]{%
  \pgfmathsetmacro{\sw}{0.12}%  slice width
  \pgfmathsetmacro{\sg}{0.06}%  gap between slices
  \pgfmathtruncatemacro{\nlast}{#4-1}%
  \pgfmathsetmacro{\totalw}{\nlast*(\sw+\sg)+\sw}%
  \pgfmathsetmacro{\hh}{0.5*#5}%
  \foreach \i in {0,...,\nlast}{%
    \pgfmathsetmacro{\xx}{#2+\i*(\sw+\sg)}%
    \filldraw[draw=blue!55!black, fill=#6, line width=0.5pt]
      (\xx,#3-\hh) rectangle ++(\sw,#5);
  }%
  \pgfmathsetmacro{\xL}{#2}%
  \pgfmathsetmacro{\xR}{#2+\totalw}%
  \pgfmathsetmacro{\xM}{0.5*(\xL+\xR)}%
  \coordinate (#1-west)  at (\xL,#3);
  \coordinate (#1-east)  at (\xR,#3);
  \coordinate (#1-north) at (\xM,#3+\hh);
  \coordinate (#1-south) at (\xM,#3-\hh);
}

\begin{document}
\begin{tikzpicture}[
  font=\small,
  conv/.style  = {-{Stealth[length=2.2mm]}, line width=1.0pt, draw=blue!80!black},
  skip/.style  = {-{Stealth[length=2.2mm]}, line width=1.0pt, draw=gray!55},
  pool/.style  = {-{Stealth[length=2.2mm]}, line width=1.0pt, draw=red!75!black},
  upc/.style   = {-{Stealth[length=2.2mm]}, line width=1.0pt, draw=green!50!black},
  outc/.style  = {-{Stealth[length=2.2mm]}, line width=1.0pt, draw=orange!80!black},
]

% --- Encoder (left arm) ---
\slicegroup{e1a}{0.0}{ 0.0}{2}{4.0}{blue!18}
\slicegroup{e1b}{0.5}{ 0.0}{2}{4.0}{blue!12}

\slicegroup{e2a}{0.9}{-2.2}{3}{3.2}{blue!18}
\slicegroup{e2b}{1.5}{-2.2}{3}{3.2}{blue!12}

\slicegroup{e3a}{1.9}{-4.2}{3}{2.4}{blue!18}
\slicegroup{e3b}{2.5}{-4.2}{3}{2.4}{blue!12}

\slicegroup{e4a}{2.9}{-5.8}{3}{1.6}{blue!18}
\slicegroup{e4b}{3.5}{-5.8}{3}{1.6}{blue!12}

% --- Bottleneck (bottom) ---
\slicegroup{bn1}{4.6}{-7.0}{3}{1.0}{blue!25}
\slicegroup{bn2}{5.2}{-7.0}{3}{1.0}{blue!18}

% --- Decoder (right arm, mirror) ---
\slicegroup{d4a}{6.3}{-5.8}{3}{1.6}{blue!8}
\slicegroup{d4b}{6.9}{-5.8}{3}{1.6}{blue!18}

\slicegroup{d3a}{7.9}{-4.2}{3}{2.4}{blue!8}
\slicegroup{d3b}{8.5}{-4.2}{3}{2.4}{blue!18}

\slicegroup{d2a}{9.5}{-2.2}{3}{3.2}{blue!8}
\slicegroup{d2b}{10.1}{-2.2}{3}{3.2}{blue!18}

\slicegroup{d1a}{11.1}{ 0.0}{2}{4.0}{blue!8}
\slicegroup{d1b}{11.6}{ 0.0}{2}{4.0}{blue!18}

% --- Output ---
\slicegroup{out}{12.4}{ 0.0}{1}{4.0}{orange!25}

% Conv arrows through blocks
\draw[conv] (e1a-east) -- (e1b-west);
\draw[conv] (e2a-east) -- (e2b-west);
\draw[conv] (e3a-east) -- (e3b-west);
\draw[conv] (e4a-east) -- (e4b-west);
\draw[conv] (bn1-east)  -- (bn2-west);
\draw[conv] (d4a-east) -- (d4b-west);
\draw[conv] (d3a-east) -- (d3b-west);
\draw[conv] (d2a-east) -- (d2b-west);
\draw[conv] (d1a-east) -- (d1b-west);

% Max pool (red, downward)
\draw[pool] (e1b-south) -- (e2a-north);
\draw[pool] (e2b-south) -- (e3a-north);
\draw[pool] (e3b-south) -- (e4a-north);
\draw[pool] (e4b-south) -- (bn1-north);

% Up-conv (green, upward)
\draw[upc] (bn2-north) -- (d4a-south);
\draw[upc] (d4b-north) -- (d3a-south);
\draw[upc] (d3b-north) -- (d2a-south);
\draw[upc] (d2b-north) -- (d1a-south);

% Skip connections (gray horizontal)
\draw[skip] (e1b-east) -- (d1a-west);
\draw[skip] (e2b-east) -- (d2a-west);
\draw[skip] (e3b-east) -- (d3a-west);
\draw[skip] (e4b-east) -- (d4a-west);

% Output arrow
\draw[outc] (d1b-east) -- (out-west);

% Labels
\node[align=center, anchor=east] at (-0.4, 0.0) {input\\image\\tile};
\node[align=left,   anchor=west] at (12.8, 0.0) {output\\segmentation\\map};
\draw[conv] (-0.3,0.0) -- (e1a-west);

% Legend
\node[anchor=north west, align=left, font=\small] at (10.0,-4.0) {%
  \tikz\draw[-{Stealth[length=2mm]},line width=1pt,draw=blue!80!black]
        (0,0)--(0.5,0); \quad Conv 3$\times$3, ReLU \\[3pt]
  \tikz\draw[-{Stealth[length=2mm]},line width=1pt,draw=gray!55]
        (0,0)--(0.5,0); \quad copy and crop \\[3pt]
  \tikz\draw[-{Stealth[length=2mm]},line width=1pt,draw=red!75!black]
        (0,-0.35)--(0,0); \quad max pool 2$\times$2 \\[3pt]
  \tikz\draw[-{Stealth[length=2mm]},line width=1pt,draw=green!50!black]
        (0,-0.35)--(0,0); \quad up conv 2$\times$2 \\[3pt]
  \tikz\draw[-{Stealth[length=2mm]},line width=1pt,draw=orange!80!black]
        (0,0)--(0.5,0); \quad conv 1$\times$1
};

\end{tikzpicture}
\end{document}

""".lstrip()


def main():
    this_file = Path(__file__)
    out_file = this_file.with_suffix(".tex")
    out_file.write_text(TEX_TEMPLATE, encoding="utf-8")


if __name__ == '__main__':
    main()
