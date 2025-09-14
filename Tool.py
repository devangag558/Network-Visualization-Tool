import sys
import subprocess

# =========================================================================
# MODIFICATION: Dependency check moved to the top to run before all other imports
# =========================================================================
try:
    from matplotlib_venn import venn2, venn3
except ImportError:
    print("Required package 'matplotlib-venn' not found. Attempting to install...")
    try:
        # Attempt to install the package
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib-venn"])
        print("\n'matplotlib-venn' installed successfully.")
        print("Please run the script again to use the new features.")
    except Exception as e:
        # Provide manual installation instructions if the automatic one fails
        print(f"\nFailed to install 'matplotlib-venn'. Please install it manually using:")
        print(f"'{sys.executable} -m pip install matplotlib-venn'")
        print(f"Error: {e}")
    sys.exit() # Exit the script after attempting installation
# =========================================================================
# END MODIFICATION
# =========================================================================

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, Counter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QVBoxLayout, QWidget,
    QPushButton, QTableWidget, QTableWidgetItem, QHBoxLayout,
    QLineEdit, QMessageBox, QSplitter, QLabel, QListWidget, QDialog,
    QTabWidget
)
from PyQt5.QtCore import Qt
import os
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import re

class VennDiagramWindow(QDialog):
    """A separate window to display the matplotlib Venn diagram."""
    def __init__(self, sets, labels, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Event Overlap Venn Diagram")
        self.setGeometry(200, 200, 800, 600)

        layout = QVBoxLayout(self)
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.plot_venn(sets, labels)

    def plot_venn(self, sets, labels):
        """Plots a Venn diagram for 2 or 3 sets with total counts in labels."""
        ax = self.figure.add_subplot(111)
        ax.clear()

        # Create new labels that include the total count for each set
        formatted_labels = []
        for label, s in zip(labels, sets):
            total_count = len(s)
            formatted_labels.append(f"{label}\n(Total: {total_count})")

        try:
            if len(sets) == 2:
                venn2(sets, set_labels=formatted_labels, ax=ax)
            elif len(sets) == 3:
                venn3(sets, set_labels=formatted_labels, ax=ax)
        except Exception as e:
            # Display error on the plot itself if something goes wrong
            ax.text(0.5, 0.5, f"Error creating diagram:\n{e}", 
                    ha='center', va='center', color='red')

        ax.set_title("Venn Diagram of Participant Overlap in Events")
        self.canvas.draw()

# =========================================================================
# MODIFICATION: The StatsWindow now uses a QTabWidget to separate stats by CSV file.
# =========================================================================
class StatsWindow(QDialog):
    """A separate window to display CSV-wise daily lab/seat statistics using tabs."""
    def __init__(self, graph_data, event_files, parent=None):
        super().__init__(parent)
        self.G = graph_data
        self.event_files = event_files
        self.setWindowTitle("CSV-wise Daily Lab/Seat Statistics")
        self.setGeometry(250, 250, 600, 500)

        layout = QVBoxLayout(self)
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        self.populate_tabs()

    def _get_seat_prefix(self, seat_str):
        """Extracts the alphabetical prefix from a seat string."""
        if not isinstance(seat_str, str):
            return "N/A"
        match = re.match(r'([a-zA-Z]+)', seat_str)
        return match.group(1) if match else "N/A"

    def populate_tabs(self):
        """Creates a tab for each CSV file with its own statistics table."""
        if not self.event_files:
            return # Should be handled by the calling method

        for file_path in self.event_files:
            # Create a widget for this tab
            tab_content_widget = QWidget()
            tab_layout = QVBoxLayout(tab_content_widget)
            table = QTableWidget()
            table.setColumnCount(3)
            table.setHorizontalHeaderLabels(["Seat Prefix", "Thursday Count", "Friday Count"])
            tab_layout.addWidget(table)

            # Get participants from this specific CSV
            try:
                df = pd.read_csv(file_path, header=None)
                participants = set(pd.concat([df[0], df[1]]).dropna().unique())
            except Exception as e:
                # If a CSV can't be read, show an error in its tab
                error_label = QLabel(f"Error reading {os.path.basename(file_path)}:\n{e}")
                self.tab_widget.addTab(error_label, os.path.basename(file_path))
                continue

            # Calculate stats for these participants
            thursday_counts = Counter()
            friday_counts = Counter()

            for roll_no in participants:
                if roll_no in self.G.nodes:
                    data = self.G.nodes[roll_no]
                    day = data.get("day")
                    seat = data.get("seat", "")
                    prefix = self._get_seat_prefix(seat)

                    if prefix == "N/A":
                        prefix = "LLLL"

                    if day == "Thursday":
                        thursday_counts[prefix] += 1
                    elif day == "Friday":
                        friday_counts[prefix] += 1

            # Populate the table for this tab
            all_prefixes = sorted(list(thursday_counts.keys() | friday_counts.keys()))

            for prefix in all_prefixes:
                thursday_count = thursday_counts.get(prefix, 0)
                friday_count = friday_counts.get(prefix, 0)
                
                row_position = table.rowCount()
                table.insertRow(row_position)
                table.setItem(row_position, 0, QTableWidgetItem(prefix))
                table.setItem(row_position, 1, QTableWidgetItem(str(thursday_count)))
                table.setItem(row_position, 2, QTableWidgetItem(str(friday_count)))
            
            table.resizeColumnsToContents()
            
            # Add the populated tab to the tab widget
            self.tab_widget.addTab(tab_content_widget, os.path.basename(file_path))
# =========================================================================
# END MODIFICATION
# =========================================================================

class NetworkGraph(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Main layout with splitter
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        # Graph canvas area
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        graph_layout = QVBoxLayout()
        graph_widget = QWidget()
        graph_layout.addWidget(self.toolbar)
        graph_layout.addWidget(self.canvas)
        graph_widget.setLayout(graph_layout)

        splitter.addWidget(graph_widget)

        # Right panel: Table + Controls
        right_panel = QVBoxLayout()
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Roll No", "Degree", "Event Files"])
        right_panel.addWidget(self.table)
        
        # Event file order display
        self.event_order_label = QLabel("Loaded Event Files (in order):")
        self.event_list_widget = QListWidget()
        self.event_list_widget.setMaximumHeight(150)
        right_panel.addWidget(self.event_order_label)
        right_panel.addWidget(self.event_list_widget)

        # Component Table
        self.components_label = QLabel("Visible Connected Components:")
        self.components_table = QTableWidget()
        self.components_table.setColumnCount(3)
        self.components_table.setHorizontalHeaderLabels(["S.No.", "Node Count", "Seat Prefixes"])
        self.components_table.setMaximumHeight(200)
        right_panel.addWidget(self.components_label)
        right_panel.addWidget(self.components_table)
        
        # Buttons layout
        button_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Event CSVs")
        self.remove_button = QPushButton("Remove Selected CSV")
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.remove_button)
        right_panel.addLayout(button_layout)
        
        analysis_layout = QHBoxLayout()
        self.export_button = QPushButton("Export Table")
        self.venn_button = QPushButton("Show Venn Diagram")
        self.stats_button = QPushButton("Show Daily Stats")
        analysis_layout.addWidget(self.export_button)
        analysis_layout.addWidget(self.venn_button)
        analysis_layout.addWidget(self.stats_button)
        right_panel.addLayout(analysis_layout)

        # Search controls
        search_layout = QHBoxLayout()
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Enter Roll No (prefix)")
        self.search_button = QPushButton("Search Roll No")
        search_layout.addWidget(self.search_box)
        search_layout.addWidget(self.search_button)
        right_panel.addLayout(search_layout)

        # View controls
        view_control_layout = QHBoxLayout()
        self.focus_button = QPushButton("Focus on Component")
        self.reset_button = QPushButton("Reset View")
        view_control_layout.addWidget(self.focus_button)
        view_control_layout.addWidget(self.reset_button)
        right_panel.addLayout(view_control_layout)

        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        splitter.addWidget(right_widget)
        
        splitter.setSizes([800, 450])

        layout.addWidget(splitter)

        # Graph data
        self.G = nx.Graph()
        self.pos = None
        self.edge_event_counts = defaultdict(lambda: defaultdict(int))
        self.event_files = []
        self.visible_components = []
        self.venn_window = None # To hold a reference to the venn window
        self.stats_window = None # To hold a reference to the stats window

        # Connections
        self.load_button.clicked.connect(self.load_event_csvs)
        self.remove_button.clicked.connect(self.remove_selected_csv)
        self.export_button.clicked.connect(self.export_table)
        self.search_button.clicked.connect(self.search_roll)
        self.reset_button.clicked.connect(self.reset_view)
        self.focus_button.clicked.connect(self.focus_on_selected_component)
        self.canvas.mpl_connect("draw_event", self.on_draw)
        self.venn_button.clicked.connect(self.show_venn_diagram)
        self.stats_button.clicked.connect(self.show_stats_window)

    def show_stats_window(self):
        """Creates and shows a new window with daily lab/seat statistics separated by CSV."""
        if not self.event_files:
            QMessageBox.warning(self, "No Data", "Please load one or more event CSV files first.")
            return

        try:
            # Pass the graph data and the list of event files to the new window
            self.stats_window = StatsWindow(self.G, self.event_files, self)
            self.stats_window.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not generate statistics window.\nError: {e}")

    def show_venn_diagram(self):
        """Creates and shows a Venn diagram for 2 or 3 loaded event files."""
        num_files = len(self.event_files)
        if not (2 <= num_files <= 3):
            QMessageBox.warning(self, "Unsupported Operation",
                                "Venn diagrams can only be generated for 2 or 3 loaded CSV files.")
            return

        sets = []
        labels = []
        try:
            for file_path in self.event_files:
                df = pd.read_csv(file_path, header=None)
                # Combine roll numbers from all columns, drop nulls, and create a unique set
                roll_numbers = pd.concat([df[0], df[1]]).dropna().unique()
                sets.append(set(roll_numbers))
                labels.append(os.path.basename(file_path))

            # Create and show the new window as a modal dialog
            self.venn_window = VennDiagramWindow(sets, labels, self)
            self.venn_window.exec_()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not generate Venn diagram.\nError: {e}")

    def focus_on_selected_component(self):
        selected_row = self.components_table.currentRow()

        if selected_row < 0:
            QMessageBox.information(self, "No Selection", "Please select a component from the table to focus on.")
            return

        if selected_row < len(self.visible_components):
            nodes_to_show = self.visible_components[selected_row]
            self.draw_graph(visible_nodes=nodes_to_show)
            
    def load_event_csvs(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Open Event CSV Files", "", "CSV Files (*.csv)")
        if not file_paths:
            return

        for path in file_paths:
            if path not in self.event_files:
                self.event_files.append(path)
        
        self.event_files.sort()
        self._rebuild_graph_and_ui()

    def remove_selected_csv(self):
        selected_item = self.event_list_widget.currentItem()
        if not selected_item:
            QMessageBox.information(self, "Information", "Please select a file to remove.")
            return
        
        selected_index = self.event_list_widget.currentRow()
        
        if 0 <= selected_index < len(self.event_files):
            self.event_files.pop(selected_index)
            self._rebuild_graph_and_ui()

    def _rebuild_graph_and_ui(self):
        self.G.clear()
        self.edge_event_counts.clear()
        
        self.event_list_widget.clear()
        for i, file_path in enumerate(self.event_files):
            self.event_list_widget.addItem(f"{i+1}: {os.path.basename(file_path)}")

        if not self.event_files:
            self.figure.clear()
            self.canvas.draw()
            self.update_table()
            return

        try:
            thursday_df = pd.read_csv("Thursday Seating.csv")
            friday_df = pd.read_csv("Friday Seating.csv")
        except FileNotFoundError as e:
            QMessageBox.critical(self, "Error", f"Seating file not found: {e.filename}. Please ensure seating files are in the same directory.")
            return

        seating_dict = {}
        for _, row in thursday_df.iterrows():
            seating_dict[row['Roll No']] = {"seat": row['Seat'], "day": "Thursday"}
        for _, row in friday_df.iterrows():
            seating_dict[row['Roll No']] = {"seat": row['Seat'], "day": "Friday"}

        for file_index, file_path in enumerate(self.event_files):
            try:
                df = pd.read_csv(file_path, header=None)
                for _, row in df.iterrows():
                    r1, r2 = row[0], row[1]
                    if pd.isna(r1) or pd.isna(r2): continue
                    
                    if r1 not in self.G.nodes:
                        seat_info = seating_dict.get(r1, {"seat": "N/A", "day": "Unknown"})
                        self.G.add_node(r1, seat=seat_info["seat"], day=seat_info["day"], events=set())
                    if r2 not in self.G.nodes:
                        seat_info = seating_dict.get(r2, {"seat": "N/A", "day": "Unknown"})
                        self.G.add_node(r2, seat=seat_info["seat"], day=seat_info["day"], events=set())
                    
                    self.G.nodes[r1]["events"].add(os.path.basename(file_path))
                    self.G.nodes[r2]["events"].add(os.path.basename(file_path))

                    self.G.add_edge(r1, r2)
                    self.edge_event_counts[tuple(sorted((r1, r2)))][file_index] += 1
            except Exception as e:
                QMessageBox.warning(self, "File Error", f"Could not process file {os.path.basename(file_path)}.\nError: {e}")
                continue

        if self.G.nodes:
            self.pos = nx.spring_layout(self.G, seed=42)
            self.update_table()
            self.draw_graph()
        else:
            self.figure.clear()
            self.canvas.draw()
            self.update_table()

    def update_table(self, visible_nodes=None):
        self.table.setRowCount(0)
        
        initial_nodes = list(visible_nodes) if visible_nodes is not None else list(self.G.nodes())
        nodes_to_display = sorted(initial_nodes, key=lambda node: (-self.G.degree(node), node))
        
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Roll No", "Degree", "Event Files"])
        
        for node in nodes_to_display:
            if node not in self.G: continue 
            
            degree = self.G.degree(node)
            events = ", ".join(sorted(list(self.G.nodes[node]["events"])))
            
            row_position = self.table.rowCount()
            self.table.insertRow(row_position)
            self.table.setItem(row_position, 0, QTableWidgetItem(str(node)))
            self.table.setItem(row_position, 1, QTableWidgetItem(str(degree)))
            self.table.setItem(row_position, 2, QTableWidgetItem(events))

    def draw_graph(self, highlight_node=None, visible_nodes=None):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if not self.G.nodes():
             self.canvas.draw()
             return

        if visible_nodes is None:
            visible_nodes = list(self.G.nodes())

        subG = self.G.subgraph(visible_nodes)

        if not subG.nodes():
            self.canvas.draw()
            return
            
        weights = dict(subG.degree())
        max_w = max(weights.values()) if weights else 1

        cmap_thursday = cm.Blues
        cmap_friday = cm.Greens

        node_colors = []
        for node in subG.nodes():
            w = weights[node] / max_w
            if subG.nodes[node]["day"] == "Thursday":
                node_colors.append(cmap_thursday(w))
            elif subG.nodes[node]["day"] == "Friday":
                node_colors.append(cmap_friday(w))
            else:
                node_colors.append((0.7, 0.7, 0.7, 1))

        node_labels = {n: f"{n}\n{subG.nodes[n]['seat']}" for n in subG.nodes()}

        nx.draw(
            subG, self.pos,
            with_labels=True, labels=node_labels,
            node_color=node_colors, node_size=2500,
            font_size=8, font_weight="bold",
            edge_color="gray", ax=ax
        )

        edge_labels = {}
        for edge in subG.edges():
            events = []
            for i in range(len(self.event_files)):
                events.append(str(self.edge_event_counts[tuple(sorted(edge))].get(i, 0)))
            edge_labels[edge] = ", ".join(events)
        nx.draw_networkx_edge_labels(subG, self.pos, edge_labels=edge_labels, font_color="red", ax=ax)

        if highlight_node and highlight_node in subG:
            nx.draw_networkx_nodes(subG, self.pos, nodelist=[highlight_node],
                                   node_color="yellow", node_size=3000, ax=ax)

        ax.set_title("Connections Between Roll Numbers with Seating Information")
        self.canvas.draw()
    
    def export_table(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Table", "", "CSV Files (*.csv)")
        if not path:
            return

        data = []
        nodes_in_table = {int(self.table.item(row, 0).text()) for row in range(self.table.rowCount())}

        for row in range(self.table.rowCount()):
            roll_no_str = self.table.item(row, 0).text()
            degree_str = self.table.item(row, 1).text()
            events = self.table.item(row, 2).text()
            roll_no = int(roll_no_str)

            connected_nodes_in_view = []
            connection_details = []

            for neighbor in self.G.neighbors(roll_no):
                if neighbor in nodes_in_table:
                    connected_nodes_in_view.append(str(neighbor))
                    
                    edge_key = tuple(sorted((roll_no, neighbor)))
                    counts_dict = self.edge_event_counts.get(edge_key, {})
                    counts_list = [str(counts_dict.get(i, 0)) for i in range(len(self.event_files))]
                    edge_label = ",".join(counts_list)
                    
                    connection_details.append(f"{neighbor}: [{edge_label}]")

            connected_nodes_str = "; ".join(connected_nodes_in_view)
            details_str = "; ".join(connection_details)
            
            data.append([roll_no_str, degree_str, events, connected_nodes_str, details_str])
        
        columns = ["Roll No", "Degree", "Event Files", "Connected Nodes (in view)", "Connection Details"]
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(path, index=False)
        QMessageBox.information(self, "Success", f"Table successfully exported to {os.path.basename(path)}")

    def search_roll(self):
        prefix = self.search_box.text().strip()
        if not prefix:
            return

        matched_nodes = [n for n in self.G.nodes() if str(n).startswith(prefix)]

        if matched_nodes:
            start_node = matched_nodes[0]
            visible_nodes = nx.node_connected_component(self.G, start_node)
            
            self.update_table(visible_nodes=visible_nodes)
            self.draw_graph(highlight_node=start_node, visible_nodes=visible_nodes)
        else:
            QMessageBox.warning(self, "Not Found", f"No roll number starting with '{prefix}' found.")

    def reset_view(self):
        self.update_table()
        self.draw_graph()

    def _get_seat_prefix(self, seat_str):
        if not isinstance(seat_str, str):
            return "N/A"
        match = re.match(r'([a-zA-Z]+)', seat_str)
        return match.group(1) if match else "N/A"

    def update_components_table(self, visible_nodes):
        self.components_table.setRowCount(0)
        self.visible_components.clear()
        if not visible_nodes:
            return

        visible_subgraph = self.G.subgraph(visible_nodes)
        components = list(nx.connected_components(visible_subgraph))
        components.sort(key=len, reverse=True)

        self.visible_components = components

        for i, component_nodes in enumerate(self.visible_components):
            node_count = len(component_nodes)
            
            prefixes_counter = Counter()
            for node in component_nodes:
                seat = self.G.nodes[node].get("seat", "")
                prefix = self._get_seat_prefix(seat)
                if prefix != "N/A":
                    prefixes_counter[prefix] += 1
            
            if not prefixes_counter:
                prefixes_str = f"LLLL - {node_count}"
            else:
                sorted_prefixes = sorted(prefixes_counter.items())
                prefix_parts = [f"{prefix} - {count}" for prefix, count in sorted_prefixes]

                counted_nodes = sum(prefixes_counter.values())
                llll_count = node_count - counted_nodes

                if llll_count > 0:
                    prefix_parts.append(f"LLLL - {llll_count}")
                
                prefixes_str = ", ".join(prefix_parts)

            row_position = self.components_table.rowCount()
            self.components_table.insertRow(row_position)
            self.components_table.setItem(row_position, 0, QTableWidgetItem(str(i + 1)))
            self.components_table.setItem(row_position, 1, QTableWidgetItem(str(node_count)))
            self.components_table.setItem(row_position, 2, QTableWidgetItem(prefixes_str))

    def on_draw(self, event):
        if not self.G or self.pos is None or not self.figure.get_axes():
            self.update_components_table([])
            return
            
        ax = self.figure.gca()
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        
        visible_nodes = [n for n, (x, y) in self.pos.items() if n in self.G and xlim[0] <= x <= xlim[1] and ylim[0] <= y <= ylim[1]]
        
        self.update_table(visible_nodes=visible_nodes)
        self.update_components_table(visible_nodes=visible_nodes)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Network Graph and Venn Diagram Viewer")
        self.setGeometry(100, 100, 1600, 900)
        self.network_graph = NetworkGraph(self)
        self.setCentralWidget(self.network_graph)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())