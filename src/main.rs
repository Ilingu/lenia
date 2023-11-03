use ahash::AHashSet;
use gfx_device_gl::{CommandBuffer, Resources};
use gfx_graphics::GfxGraphics;
use piston_window::*;

const CELL_DIMENSION: f64 = 2.0;

const DEFAULT_WIDTH: u32 = 1500;
const DEFAULT_HEIGHT: u32 = 900;

const DEFAULT_WCELL_COUNT: usize = DEFAULT_WIDTH as usize / CELL_DIMENSION as usize;
const DEFAULT_HCELL_COUNT: usize = DEFAULT_HEIGHT as usize / CELL_DIMENSION as usize;

enum Mode {
    Lenia,
    GameOfLife,
}

struct Lenia {
    cells: Vec<Vec<f32>>,
    active_cells: AHashSet<(usize, usize)>,
    mode: Mode,
    delta_t: f64,
}

impl Lenia {
    fn new(
        (wcell_count, hcell_count): (usize, usize),
        ((area_w_min, mut area_w_max), (area_h_min, mut area_h_max)): (
            (usize, usize),
            (usize, usize),
        ),
        mode: Option<Mode>,
        delta_t: Option<f64>,
    ) -> Self {
        if area_w_max >= wcell_count {
            area_w_max = wcell_count - 1;
        }
        if area_h_max >= hcell_count {
            area_h_max = hcell_count - 1;
        }

        let mut cells = vec![vec![0_f32; wcell_count]; hcell_count];
        for raw in cells.iter_mut().take(area_h_max + 1).skip(area_h_min) {
            for cell in raw.iter_mut().take(area_w_max + 1).skip(area_w_min) {
                *cell = match mode.as_ref().unwrap_or(&Mode::Lenia) {
                    Mode::Lenia => fastrand::f32(),
                    Mode::GameOfLife => fastrand::usize(0..=1) as f32,
                }
            }
        }

        Self {
            cells,
            active_cells: AHashSet::new(),
            mode: mode.unwrap_or(Mode::Lenia),
            delta_t: delta_t.unwrap_or(1.0),
        }
    }

    fn resize(&mut self, new_cell_width_count: usize, new_cell_height_count: usize) {
        if new_cell_width_count != self.cells[0].len() {
            for raw in 0..(self.cells.len()) {
                self.cells[raw].resize(new_cell_width_count, 0.0);
            }
        }
        if new_cell_height_count != self.cells.len() {
            self.cells
                .resize(new_cell_height_count, vec![0.0; new_cell_width_count]);
        }
    }

    fn compute_next_frame(&mut self) {
        match self.mode {
            Mode::Lenia => self.compute_next_lenia_frame(),
            Mode::GameOfLife => self.compute_next_gol_frame(),
        }
    }

    fn compute_next_lenia_frame(&mut self) {}

    fn compute_next_gol_frame(&mut self) {
        let (w, h) = (self.cells[0].len(), self.cells.len());

        let mut next_frame_cells = self.cells.clone();
        let mut next_frame_active_cells: AHashSet<(usize, usize)> = AHashSet::new();

        let mut update_cell = |raw: usize, col: usize| {
            let (top_raw, bottom_raw) = (
                if raw == 0 { h - 1 } else { raw - 1 },
                if raw == h - 1 { 0 } else { raw + 1 },
            );
            let (left_col, right_col) = (
                if col == 0 { w - 1 } else { col - 1 },
                if col == w - 1 { 0 } else { col + 1 },
            );

            let neighbours = [
                self.cells[top_raw][left_col],     // top left
                self.cells[top_raw][col],          // top mid
                self.cells[top_raw][right_col],    // top right
                self.cells[raw][left_col],         // mid left
                self.cells[raw][right_col],        // mid right
                self.cells[bottom_raw][left_col],  // bottom left
                self.cells[bottom_raw][col],       // bottom mid
                self.cells[bottom_raw][right_col], // bottom right
            ];

            let is_alive = self.cells[raw][col] == 1.0;
            let alive_cells_count = neighbours.into_iter().sum::<f32>() as usize;

            if (is_alive && (2..=3).contains(&alive_cells_count))
                || (!is_alive && alive_cells_count == 3)
            {
                next_frame_cells[raw][col] = 1.0;
            } else {
                next_frame_cells[raw][col] = 0.0;
            }

            // change detected, add all affected cells (neighbours and current cells)
            if self.cells[raw][col] != next_frame_cells[raw][col] {
                next_frame_active_cells.insert((raw, col));
                next_frame_active_cells.insert((top_raw, left_col));
                next_frame_active_cells.insert((top_raw, col));
                next_frame_active_cells.insert((top_raw, right_col));
                next_frame_active_cells.insert((raw, left_col));
                next_frame_active_cells.insert((raw, right_col));
                next_frame_active_cells.insert((bottom_raw, left_col));
                next_frame_active_cells.insert((bottom_raw, col));
                next_frame_active_cells.insert((bottom_raw, right_col));
            }
        };

        if self.active_cells.is_empty() {
            #[allow(clippy::needless_range_loop)]
            for raw in 0..self.cells.len() {
                for col in 0..w {
                    update_cell(raw, col);
                }
            }
        } else {
            for &(raw, col) in &self.active_cells {
                update_cell(raw, col);
            }
        }

        self.cells = next_frame_cells;
        self.active_cells = next_frame_active_cells;
    }

    fn render(&self, context: Context, graphics: &mut GfxGraphics<'_, Resources, CommandBuffer>) {
        let w = self.cells[0].len();
        for raw in 0..self.cells.len() {
            for col in 0..w {
                rectangle(
                    [1.0, 1.0, 1.0, self.cells[raw][col]], // red
                    [
                        col as f64 * CELL_DIMENSION,
                        raw as f64 * CELL_DIMENSION,
                        CELL_DIMENSION,
                        CELL_DIMENSION,
                    ],
                    context.transform,
                    graphics,
                );
            }
        }
    }
}

fn generate_spawn_area(wcell_count: usize, hcell_count: usize) -> ((usize, usize), (usize, usize)) {
    (
        {
            let area_w_min = fastrand::usize(0..wcell_count);
            (
                area_w_min,
                area_w_min + fastrand::usize(1..(wcell_count - area_w_min)),
            )
        },
        {
            let area_h_min = fastrand::usize(0..hcell_count);
            (
                area_h_min,
                area_h_min + fastrand::usize(1..(hcell_count - area_h_min)),
            )
        },
    )
}

struct AppState {
    is_game_paused: bool,
    is_drawing: bool,
    is_erasing: bool,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            is_game_paused: true,
            is_drawing: false,
            is_erasing: false,
        }
    }
}

fn main() {
    let mut lenia = Lenia::new(
        (DEFAULT_WCELL_COUNT, DEFAULT_HCELL_COUNT),
        generate_spawn_area(DEFAULT_WCELL_COUNT, DEFAULT_HCELL_COUNT),
        Some(Mode::GameOfLife),
        None,
    );
    let mut window: PistonWindow = WindowSettings::new("Lenia!", [DEFAULT_WIDTH, DEFAULT_HEIGHT])
        .build()
        .unwrap();
    // window.set_max_fps(12);

    let mut app_state = AppState::default();
    while let Some(event) = window.next() {
        let Size { width, height } = window.size();
        let (wcell_count, hcell_count) = (
            (width / CELL_DIMENSION) as usize,
            (height / CELL_DIMENSION) as usize,
        );
        lenia.resize(wcell_count, hcell_count);

        if let Event::Input(input, _) = &event {
            match input {
                Input::Move(Motion::MouseCursor([x, y])) => {
                    if app_state.is_drawing {
                        let (raw, col) = (
                            (y / CELL_DIMENSION).floor() as usize,
                            (x / CELL_DIMENSION).floor() as usize,
                        );
                        lenia.cells[raw][col] = if app_state.is_erasing { 0.0 } else { 1.0 };
                    }
                }
                Input::Text(text) => {
                    let character = text.chars().next().unwrap(); // cannot panic
                    match character {
                        'r' => {
                            let ((area_w_min, area_w_max), (area_h_min, area_h_max)) =
                                generate_spawn_area(wcell_count, hcell_count);
                            for raw in lenia.cells.iter_mut().take(area_h_max + 1).skip(area_h_min)
                            {
                                for cell in raw.iter_mut().take(area_w_max + 1).skip(area_w_min) {
                                    *cell = match lenia.mode {
                                        Mode::Lenia => fastrand::f32(),
                                        Mode::GameOfLife => fastrand::usize(0..=1) as f32,
                                    }
                                }
                            }
                        }
                        'c' => {
                            lenia.cells = vec![vec![0_f32; lenia.cells[0].len()]; lenia.cells.len()]
                        }
                        's' => {
                            app_state.is_game_paused = false;
                            app_state.is_drawing = false;
                        }
                        'h' => app_state.is_game_paused = true,
                        'd' => {
                            app_state.is_drawing = !app_state.is_drawing;
                            app_state.is_game_paused = true;
                        }
                        _ => (),
                    }
                }
                Input::Button(button_action) => {
                    if app_state.is_drawing {
                        if let Button::Mouse(mouse_action) = button_action.button {
                            match mouse_action {
                                MouseButton::Left => app_state.is_erasing = false,
                                MouseButton::Right => app_state.is_erasing = true,
                                _ => (),
                            }
                        }
                    }
                }
                _ => (),
            }
        }

        window.draw_2d(&event, |context, graphics, _device| {
            clear([0.0, 0.0, 0.0, 1.0], graphics);
            if !app_state.is_game_paused {
                lenia.compute_next_frame();
            }
            lenia.render(context, graphics);
        });
    }
}
