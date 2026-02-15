from nicegui import ui, app
import os
import json
import sys
from pathlib import Path
from tkinter import Tk, filedialog
import queue

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from GQA_cluster_transformer.llama_cluster_transformer import llama_cluster_transformer, Tokenizer, RMSNorm
from GQA_cluster_transformer.transformer_model_handler import hugging_face_model_handler

CLUSTER_MATRIX_DIR = os.path.join(PROJECT_ROOT, "cluster_matrix")
if CLUSTER_MATRIX_DIR not in sys.path:
    sys.path.insert(0, CLUSTER_MATRIX_DIR)

from cluster_matrix_v1 import cluster_matrix
from cluster_matrix_v1 import cluster_zmq
from cluster_matrix_v1 import check_combined_result_values



import threading


class ClusterTransformerUI:

    def __init__(self, tokenizer=None, model=None):
        os.environ["CLUSTER_FORCE_FP32"] = "1"
        # -----------------------------
        # State
        # -----------------------------
        self.messages = []

        self.IP_list = []
        self.percentages = []
        self.CPU_GPU_select_list = []
        self.backend_select_list = []

        self.selected_model_path = None

        self.cluster_transformer = None

        self.Tokenizer = None
        self.ui_model = None
        self._pending_path = None
        self.config_dir = PROJECT_ROOT / 'GUI' / 'cluster_configs'
        self.model_busy = False
        self.model_action = None
        self.model_error = None
        self.node_widgets = []
        self.msg_counter = 0
        self.model_ready = False
        self.system = "You are a helpful assistant."
        self.special_token_ids = set()

        self.max_gen_len = 128
        self.temperature = 0.8
        self.top_p = 0.95
        # -----------------------------
        # Build UI
        # -----------------------------
        self.build_ui()

    # =========================================================
    # Actions
    # =========================================================

    def init_cluster_model(self, saveOrLoad = 'save'):
        if not self.IP_list:
            ui.notify('Add at least one node before saving/loading.', type='warning')
            return
        if not self.selected_model_path:
            ui.notify('Select a model folder first.', type='warning')
            return
        if self.model_busy:
            ui.notify('Model task already running.', type='warning')
            return

        self.model_busy = True
        self.model_action = saveOrLoad
        self.model_error = None
        if hasattr(self, 'progress_bar'):
            self.progress_bar.value = 0
            self.progress_bar.update()
        if hasattr(self, 'progress_label'):
            self.progress_label.set_text(f'{saveOrLoad.title()} in progress...')

        # Apply generation defaults from generation_config.json (if available).
        self._apply_model_generation_settings(self.selected_model_path)

        def _run():
            try:
                cluster_zmq_obj = cluster_zmq(self.IP_list)
                self.ui_model = hugging_face_model_handler(
                    model_path=self.selected_model_path,
                    cluster_zmq_object=cluster_zmq_obj,
                    percentages=self.percentages,
                    CPU_GPU_select_list=self.CPU_GPU_select_list,
                    backend_select_list=self.backend_select_list,
                )

                self.ui_model.cache_model_tensors(saveOrload=saveOrLoad)
                self.Tokenizer = Tokenizer(self.selected_model_path)
                self.cluster_transformer = llama_cluster_transformer(self.Tokenizer, self.ui_model)
                self.cluster_transformer.system = self.system
                model_specials = getattr(self.ui_model, 'special_token_ids', None)
                if model_specials:
                    self.special_token_ids = set(int(x) for x in model_specials if isinstance(x, int))
                self.model_error = None
                self.model_ready = True
            except Exception as exc:
                self.model_error = str(exc)
                self.model_ready = False
            finally:
                self.model_busy = False

        threading.Thread(target=_run, daemon=True).start()

    def choose_folder(self):
        def _open_dialog():
            root = Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            root.lift()
            root.focus_force()
            root.update()
            path = filedialog.askdirectory()
            root.destroy()
            if path:
                self._pending_path = path

        threading.Thread(target=_open_dialog, daemon=True).start()

    def _consume_pending_path(self):
        if self._pending_path:
            self.selected_model_path = self._pending_path
            self.selected_path_label.set_text(self._pending_path)
            self._pending_path = None

    def _update_progress(self):
        if not hasattr(self, 'progress_bar'):
            return
        if self.model_error and not self.model_busy:
            self.progress_label.set_text(f'Error: {self.model_error}')
            return
        if not self.ui_model or not self.model_action:
            return
        total = getattr(self.ui_model, 'num_layers', 0) or 0
        if total <= 0:
            total = getattr(self.ui_model, 'num_hidden_layers', 0) or 0
        if total <= 0:
            self.progress_label.set_text(f'{self.model_action.title()} in progress...')
            return
        if self.model_action == 'save':
            current = getattr(self.ui_model, 'save_progress', 0)
        else:
            current = getattr(self.ui_model, 'load_progress', 0)
        value = max(0.0, min(float(current) / float(total), 1.0))
        self.progress_bar.value = value
        self.progress_bar.update()
        self.progress_label.set_text(
            f'{self.model_action.title()} {current}/{total}'
        )
        if not self.model_busy and value >= 1.0:
            self.progress_label.set_text(f'{self.model_action.title()} complete')
            self.model_action = None

    def send_message(self):
        text = self.user_input.value.strip()
        if not text:
            return
        if not self.model_ready or not self.cluster_transformer:
            ui.notify('Load or save a model first, then try again.', type='warning')
            return

        bubble_id = f'assistant-bubble-{self.msg_counter}'
        self.msg_counter += 1

        with self.chat_container:
            with ui.element('div').classes('chat-row user'):
                ui.label(text).classes('chat-bubble user')
            with ui.element('div').classes('chat-row assistant'):
                assistant_label = ui.label('').classes('chat-bubble assistant').props(f'id={bubble_id}')

        self.user_input.value = ''
        ui.run_javascript(
            "const c=document.querySelector('.chat-panel'); if(c){c.scrollTop=c.scrollHeight;}"
        )

        self.messages.append(('user', text))
        token_queue = queue.Queue()
        stream_state = {'done': False, 'error': None, 'text': ''}

        def on_token(batch, token_id, token_text):
            if batch != 0:
                return
            try:
                if token_id in self.special_token_ids:
                    return
            except Exception:
                pass
            text_piece = token_text or ''
            try:
                sp = self.cluster_transformer.tokenizer.sp_model
                piece = sp.id_to_piece(int(token_id))
                if piece in ('<s>', '</s>', '<unk>'):
                    text_piece = ''
                elif piece.startswith('▁'):
                    text_piece = ' ' + piece[1:]
                else:
                    text_piece = piece
                text_piece = text_piece.replace('▁', ' ')
                text_piece = text_piece.replace('<0x0A>', '\n')
            except Exception:
                # Fallback: strip known special tokens when using HF tokenizers.
                specials = [
                    getattr(self.cluster_transformer, 'eos_token_text', None),
                    getattr(self.cluster_transformer, 'eot_token_text', None),
                    getattr(self.cluster_transformer, 'bos_token_text', None),
                    getattr(self.cluster_transformer, 'pad_token_text', None),
                ]
                if text_piece in [s for s in specials if isinstance(s, str)]:
                    text_piece = ''
            if text_piece:
                text_piece = text_piece.replace('<0x0A>', '\n')
            if text_piece:
                token_queue.put(text_piece)

        def _run_generation():
            try:
                self.cluster_transformer.generate(
                    list(self.messages),
                    max_gen_len=self.max_gen_len,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    on_token=on_token,
                )
            except Exception as exc:
                stream_state['error'] = str(exc)
            finally:
                stream_state['done'] = True

        threading.Thread(target=_run_generation, daemon=True).start()

        def _flush_tokens():
            updated = False
            while not token_queue.empty():
                stream_state['text'] += token_queue.get()
                updated = True
            if updated:
                assistant_label.set_text(stream_state['text'])
                assistant_label.update()
                ui.run_javascript(
                    "const c=document.querySelector('.chat-panel'); if(c){c.scrollTop=c.scrollHeight;}"
                )
            if stream_state['done'] and token_queue.empty():
                if stream_state['error']:
                    assistant_label.set_text(f"Error: {stream_state['error']}")
                    assistant_label.update()
                else:
                    if stream_state['text']:
                        self.messages.append(('assistant', stream_state['text']))
                token_timer.cancel()

        token_timer = ui.timer(0.05, _flush_tokens)

    def _set_backend(self, idx, value):
        if 0 <= idx < len(self.backend_select_list):
            self.backend_select_list[idx] = value

    def _set_percentage(self, idx, value):
        try:
            pct = float(value)
        except (TypeError, ValueError):
            return
        if 0 <= idx < len(self.percentages):
            self.percentages[idx] = pct

    def _set_cpu_gpu(self, idx, value):
        if 0 <= idx < len(self.CPU_GPU_select_list):
            self.CPU_GPU_select_list[idx] = (value == 'GPU')

    def _event_value(self, e):
        if hasattr(e, 'value'):
            return e.value
        if hasattr(e, 'args') and isinstance(e.args, dict) and 'value' in e.args:
            return e.args['value']
        if hasattr(e, 'sender') and hasattr(e.sender, 'value'):
            return e.sender.value
        return None

    def _read_json(self, path: str):
        try:
            if not path or not os.path.exists(path):
                return None
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None

    def _apply_model_generation_settings(self, model_dir: str):
        if not model_dir:
            return
        gen = self._read_json(os.path.join(model_dir, 'generation_config.json')) or {}
        temp = gen.get('temperature', None)
        top_p = gen.get('top_p', None)
        if temp is not None:
            try:
                self.temperature = float(temp)
            except (TypeError, ValueError):
                pass
        if top_p is not None:
            try:
                self.top_p = float(top_p)
            except (TypeError, ValueError):
                pass

        # Update UI inputs if they exist
        if hasattr(self, 'temperature_input'):
            self.temperature_input.value = str(self.temperature)
            self.temperature_input.update()
        if hasattr(self, 'top_p_input'):
            self.top_p_input.value = str(self.top_p)
            self.top_p_input.update()

    def _set_system(self, value):
        if value is None:
            return
        self.system = str(value)
        if self.cluster_transformer is not None and hasattr(self.cluster_transformer, 'system'):
            self.cluster_transformer.system = self.system

    def _add_node_ui(self, ip, percentage=0.1, backend='llama', device='GPU'):
        self.IP_list.append(ip)
        self.percentages.append(float(percentage))
        self.CPU_GPU_select_list.append(device == 'GPU')
        self.backend_select_list.append(backend)

        idx = len(self.IP_list) - 1
        with self.node_config_container:
            ui.separator()
            with ui.column().classes('w-full gap-2 items-stretch node-config-card'):
                ui.label(f'🧩 NODE {ip}').classes('font-bold text-sm')
                backend_select = ui.select(['llama', 'torch'], value=backend, label='Backend').props('outlined').classes('w-full').on(
                    'change', lambda e, i=idx: self._set_backend(i, e.value)
                )
                percentage_input = ui.input(value=str(percentage), label='Percentage').props('outlined type=number step=0.05').classes('w-full').on(
                    'change', lambda e, i=idx: self._set_percentage(i, e.value)
                )
                device_select = ui.select(['GPU', 'CPU'], value=device, label='Device').props('outlined').classes('w-full').on(
                    'change', lambda e, i=idx: self._set_cpu_gpu(i, e.value)
                )
        self.node_widgets.append({
            'backend': backend_select,
            'percentage': percentage_input,
            'device': device_select,
        })

    def add_node(self):
        ip = self.ip_input.value.strip()
        if not ip:
            return

        self._add_node_ui(ip)
        self.ip_input.value = ''
        ui.notify(f'Node {ip} added')

    def _list_config_names(self):
        if not self.config_dir.exists():
            return []
        return sorted(
            p.stem for p in self.config_dir.glob('*.json') if p.is_file()
        )

    def _refresh_config_list(self):
        if hasattr(self, 'config_select'):
            self.config_select.options = self._list_config_names()
            self.config_select.update()

    def save_node_config(self):
        name = (self.config_name_input.value or '').strip()
        if not name:
            ui.notify('Enter a config name first.', type='warning')
            return
        if not self.IP_list:
            ui.notify('Add at least one node before saving.', type='warning')
            return
        self.config_dir.mkdir(parents=True, exist_ok=True)
        nodes = []
        for i, ip in enumerate(self.IP_list):
            backend = self.backend_select_list[i] if i < len(self.backend_select_list) else 'llama'
            pct = self.percentages[i] if i < len(self.percentages) else 0.1
            device = 'GPU' if (self.CPU_GPU_select_list[i] if i < len(self.CPU_GPU_select_list) else True) else 'CPU'
            if i < len(self.node_widgets):
                w = self.node_widgets[i]
                backend = w['backend'].value or backend
                device = w['device'].value or device
                try:
                    pct = float(w['percentage'].value)
                except (TypeError, ValueError):
                    pass
            nodes.append({
                'ip': ip,
                'percentage': float(pct),
                'device': device,
                'backend': backend,
            })
        payload = {
            'name': name,
            'nodes': nodes,
        }
        path = self.config_dir / f'{name}.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        self._refresh_config_list()
        ui.notify(f'Saved config: {name}')

    def load_node_config(self):
        name = (self.config_select.value or '').strip()
        if not name:
            ui.notify('Select a config to load.', type='warning')
            return
        path = self.config_dir / f'{name}.json'
        if not path.exists():
            ui.notify('Config file not found.', type='warning')
            return
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.IP_list = []
        self.percentages = []
        self.CPU_GPU_select_list = []
        self.backend_select_list = []
        self.node_config_container.clear()
        self.node_widgets = []

        nodes = data.get('nodes', [])
        for node in nodes:
            ip = node.get('ip', '')
            pct = node.get('percentage', 0.1)
            try:
                pct = float(pct)
            except (TypeError, ValueError):
                pct = 0.1
            backend = node.get('backend', 'llama')
            device = node.get('device', 'GPU')
            if ip:
                self._add_node_ui(ip, pct, backend, device)

        # Ensure UI widgets reflect loaded values
        for i, node in enumerate(nodes):
            if i < len(self.node_widgets):
                w = self.node_widgets[i]
                w['backend'].value = node.get('backend', 'llama')
                w['backend'].update()
                w['percentage'].value = str(node.get('percentage', 0.1))
                w['percentage'].update()
                w['device'].value = node.get('device', 'GPU')
                w['device'].update()

        # Sync internal lists with loaded values
        for i, node in enumerate(nodes):
            if i < len(self.IP_list):
                try:
                    self.percentages[i] = float(node.get('percentage', 0.1))
                except (TypeError, ValueError):
                    self.percentages[i] = 0.1
                self.backend_select_list[i] = node.get('backend', 'llama')
                self.CPU_GPU_select_list[i] = (node.get('device', 'GPU') == 'GPU')

        ui.notify(f'Loaded config: {name}')

    def build_ui(self):
        # -----------------------------
        # Static assets + CSS
        # -----------------------------
        app.add_static_files('/assets', str(PROJECT_ROOT))
        ui.add_head_html('''
        <style>
        :root {
            --title-glow: #7fffd4;
            --title-edge: #00c2ff;
        }
        body {
            background:
                linear-gradient(rgba(0, 0, 0, 0.45), rgba(0, 0, 0, 0.45)),
                radial-gradient(1200px 800px at 20% 10%, rgba(0, 255, 200, 0.10), transparent 60%),
                radial-gradient(1000px 700px at 80% 0%, rgba(0, 140, 255, 0.12), transparent 55%),
                url("/assets/e677e89c-b7ca-40d4-963e-414d37f2a97b.png");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            background-repeat: no-repeat;
        }
        .glass {
            background: rgba(20, 20, 20, 0.12);
            border-radius: 12px;
        }
        .main-content {
            width: calc(100% - 24rem);
            margin-right: 24rem;
            box-sizing: border-box;
        }
        @media (max-width: 1024px) {
            .main-content {
                width: 100%;
                margin-right: 0;
            }
        }
        .image-frame {
            width: 16rem;
            height: 16rem;
            padding: 0;
            overflow: hidden;
            border-radius: 12px;
            background: rgba(0, 0, 0, 0.05);
        }
        .title-glow {
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            text-shadow:
                0 0 6px var(--title-glow),
                0 0 24px var(--title-edge),
                0 0 48px rgba(0, 194, 255, 0.4);
        }
        .title-box {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 14px;
            background: rgba(15, 15, 15, 0.35);
            backdrop-filter: blur(6px);
            -webkit-backdrop-filter: blur(6px);
        }
        .chat-row {
            display: flex;
            width: 100%;
        }
        .chat-row.user {
            justify-content: flex-end;
        }
        .chat-row.assistant {
            justify-content: flex-start;
        }
        .chat-panel {
            width: 100%;
            max-width: 100%;
            margin: 0;
        }
        .chat-bubble {
            max-width: 80%;
            padding: 10px 14px;
            border-radius: 14px;
            font-size: 0.95rem;
            line-height: 1.35;
            white-space: pre-wrap;
            background: #141414;
            border: 1px solid rgba(255, 255, 255, 0.15);
        }
        .chat-bubble.user {
            background: #00c88c;
            border: 1px solid #0fe0a0;
        }
        .chat-bubble.assistant {
            background: #ffb34a;
            border: 1px solid #ffc46b;
        }
        .input-bar {
            background: rgba(255, 255, 255, 0.08);
            border-top: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 12px;
        }
        .chat-input input {
            background: rgba(255, 255, 255, 0.92) !important;
            color: #111 !important;
            border-radius: 10px !important;
        }
        .node-config-card {
            padding: 10px;
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        </style>
        ''')

        # -----------------------------
        # Right drawer
        # -----------------------------
        with ui.right_drawer().props('overlay=false').classes('glass w-96'):

            with ui.tabs().classes('w-full') as tabs:
                model_config_tab = ui.tab('model config')
                cluster_config_tab = ui.tab('cluster config')

            # -------- Cluster config --------
            with ui.tab_panels(tabs, value=cluster_config_tab).classes('w-full'):
                with ui.tab_panel(cluster_config_tab).classes('w-full'):

                    with ui.column().classes('w-full'):
                        ui.label('🖧 Cluster Nodes').classes('text-lg font-bold w-full text-center')
                        ui.separator()

                        self.config_name_input = ui.input(
                            placeholder='Config name'
                        ).classes('w-full')
                        with ui.row().classes('w-full gap-2'):
                            ui.button('💾 Save Config', on_click=self.save_node_config).classes('flex-1')
                            ui.button('📥 Load Config', on_click=self.load_node_config).classes('flex-1')

                        self.config_select = ui.select(
                            self._list_config_names(),
                            label='Saved configs'
                        ).classes('w-full')

                        ui.separator()

                        self.ip_input = ui.input(
                            placeholder='Node IP (e.g. 192.168.2.100)'
                        ).classes('w-full')

                        ui.button('➕ Add node', on_click=self.add_node).classes('w-full')

                        ui.separator()
                        ui.label('⚙️ Node Config').classes('text-lg font-bold w-full')

                        self.node_config_container = ui.column().classes('w-full gap-2 items-stretch')

            # -------- Model config --------
            with ui.tab_panels(tabs, value=model_config_tab):
                with ui.tab_panel(model_config_tab):

                    ui.label('Model config setting')

                    with ui.column().classes('w-full gap-2 p-2'):
                        with ui.row().classes('w-full gap-2'):
                            ui.button('💾 Save', on_click=lambda: self.init_cluster_model('save')).classes('flex-1')
                            ui.button('📂 Load', on_click=lambda: self.init_cluster_model('load')).classes('flex-1')
                            ui.button('📁 Select Folder', on_click=self.choose_folder).classes('flex-1')

                        with ui.row().classes('w-full gap-2'):
                            ui.input(value=str(self.max_gen_len), label='Max Gen Len').props('outlined type=number step=1 min=1').classes('flex-1').on(
                                'change', lambda e: setattr(self, 'max_gen_len', int(float(self._event_value(e))) if self._event_value(e) not in (None, '') else self.max_gen_len)
                            )
                            self.temperature_input = ui.input(value=str(self.temperature), label='Temperature').props('outlined type=number step=0.05 min=0').classes('flex-1').on(
                                'change', lambda e: setattr(self, 'temperature', float(self._event_value(e)) if self._event_value(e) not in (None, '') else self.temperature)
                            )
                            self.top_p_input = ui.input(value=str(self.top_p), label='Top P').props('outlined type=number step=0.01 min=0 max=1').classes('flex-1').on(
                                'change', lambda e: setattr(self, 'top_p', float(self._event_value(e)) if self._event_value(e) not in (None, '') else self.top_p)
                            )

                        ui.input(value=self.system, label='System Prompt').props('outlined').classes('w-full').on(
                            'change', lambda e: self._set_system(self._event_value(e))
                        )

                        self.progress_label = ui.label('Idle').classes('text-sm')
                        self.progress_bar = ui.linear_progress(value=0).props('instant-feedback').classes('w-full')

                    ui.separator()
                    self.selected_path_label = ui.label('No folder selected')

        # -----------------------------
        # Main page
        # -----------------------------
        with ui.column().classes('main-content w-full h-screen p-4 gap-4 items-stretch'):

            with ui.row().classes('w-full items-center'):
                ui.element('div').classes('w-64')
                with ui.element('div').classes('title-box'):
                    ui.label('Open Cluster AI Station Beta').classes('title-glow')
                with ui.element('div').classes('w-64'):
                    with ui.element('div').classes('image-frame'):
                        self.ui_image = ui.image('e677e89c-b7ca-40d4-963e-414d37f2a97b.png').style(
                            'width:100%;height:100%;object-fit:cover;opacity:0.8;'
                        )

            # ---------- Chat area ----------
            self.chat_container = ui.column().classes(
                'chat-panel w-full flex-grow overflow-y-auto p-4 gap-2 items-start justify-start'
            )

            # ---------- Input bar ----------
            with ui.row().classes('w-full p-4 items-center input-bar'):
                self.user_input = ui.input(
                    placeholder='Type your message...'
                ).classes('flex-grow chat-input')

                ui.button('Send', on_click=self.send_message)
                self.user_input.on('keydown.enter', self.send_message)

        ui.timer(0.5, self._consume_pending_path)
        ui.timer(0.2, self._update_progress)


# =========================================================
# Main entry point
# =========================================================

def main():
    ClusterTransformerUI()
    ui.run(title='Open Cluster AI Station beta')



main()
