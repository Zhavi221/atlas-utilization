"""
Statistics plotter for pipeline data.

Generates 4 separate visualization categories:
1. Pipeline Performance   – timing, throughput, memory, success rate
2. Parsed Data Overview   – total events, chunks, particle breakdown
3. Invariant Mass Summary – final states, combinations, objects
4. Histogram Metadata     – histograms created, peaks cut, main vs outlier
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Any
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for servers
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.patches import FancyBboxPatch


class StatisticsPlotter:
    """
    Creates separated statistical plot categories from pipeline data.
    """

    # Consistent color palette
    COLORS = {
        'success': '#27ae60',
        'failure': '#e74c3c',
        'primary': '#2980b9',
        'secondary': '#8e44ad',
        'accent': '#f39c12',
        'info': '#16a085',
        'electrons': '#3498db',
        'muons': '#e74c3c',
        'jets': '#2ecc71',
        'photons': '#f1c40f',
        'bg_light': '#ecf0f1',
        'text_dark': '#2c3e50',
        'grid': '#bdc3c7',
    }

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        plt.style.use('seaborn-v0_8-whitegrid')

    # =========================================================================
    # PLOT 1: Pipeline Performance
    # =========================================================================
    def plot_pipeline_performance(
        self,
        parsing_stats: Optional[dict] = None,
        mass_calc_stats: Optional[dict] = None,
        post_proc_stats: Optional[dict] = None,
        hist_stats: Optional[dict] = None,
        save_name: str = "01_pipeline_performance.png",
    ) -> Optional[Path]:
        """
        Pipeline-level performance metrics:
        - Execution time per stage (stacked bar)
        - Throughput (events/sec)
        - Memory usage
        - Overall success rate donut
        - File processing waterfall
        - Stage timing breakdown
        """
        self.logger.info("Generating pipeline performance plot...")

        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('Pipeline Performance Dashboard', fontsize=20, fontweight='bold',
                     color=self.COLORS['text_dark'], y=0.98)

        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

        # -- 1a. Stage Execution Times --
        ax1 = fig.add_subplot(gs[0, 0])
        stages = []
        times = []
        colors_stage = []
        stage_map = [
            ('Parsing', parsing_stats, self.COLORS['primary']),
            ('Mass Calc', mass_calc_stats, self.COLORS['secondary']),
            ('Post-Proc', post_proc_stats, self.COLORS['accent']),
            ('Histograms', hist_stats, self.COLORS['info']),
        ]
        for name, stats, color in stage_map:
            if stats:
                t = self._safe_float(stats.get('total_time_sec', 0))
                stages.append(name)
                times.append(t)
                colors_stage.append(color)

        if stages:
            bars = ax1.barh(stages, times, color=colors_stage, edgecolor='white',
                           linewidth=1.5, height=0.6)
            for bar, t in zip(bars, times):
                ax1.text(bar.get_width() + max(times) * 0.02,
                         bar.get_y() + bar.get_height() / 2,
                         f'{t:.1f}s', va='center', fontweight='bold', fontsize=10)
            ax1.set_xlabel('Time (seconds)', fontsize=11)
            ax1.set_title('Stage Execution Time', fontsize=13, fontweight='bold', pad=10)
            ax1.invert_yaxis()
        else:
            self._empty_panel(ax1, 'No stage timing data')
            ax1.set_title('Stage Execution Time', fontsize=13, fontweight='bold', pad=10)

        # -- 1b. Success Rate Donut --
        ax2 = fig.add_subplot(gs[0, 1])
        total_success = 0
        total_fail = 0
        if parsing_stats:
            total_success += int(parsing_stats.get('successful_files', 0))
            total_fail += int(parsing_stats.get('failed_files', 0))

        if total_success + total_fail > 0:
            rate = total_success / (total_success + total_fail) * 100
            wedges, _, autotexts = ax2.pie(
                [total_success, total_fail],
                labels=None,
                colors=[self.COLORS['success'], self.COLORS['failure']],
                autopct='%1.1f%%', startangle=90,
                pctdistance=0.78,
                wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2),
            )
            for at in autotexts:
                at.set_fontweight('bold')
                at.set_fontsize(11)
            ax2.text(0, 0, f'{rate:.0f}%', ha='center', va='center',
                     fontsize=28, fontweight='bold', color=self.COLORS['text_dark'])
            ax2.legend(['Success', 'Failed'], loc='lower center',
                       fontsize=10, ncol=2, frameon=False)
        else:
            self._empty_panel(ax2, 'No file data')
        ax2.set_title('Overall Success Rate', fontsize=13, fontweight='bold', pad=10)

        # -- 1c. Throughput --
        ax3 = fig.add_subplot(gs[0, 2])
        throughput_data = {}
        if parsing_stats:
            t = self._safe_float(parsing_stats.get('total_time_sec', 0))
            evts = int(parsing_stats.get('total_events', 0))
            if t > 0:
                throughput_data['Parsing'] = evts / t

        if throughput_data:
            names = list(throughput_data.keys())
            vals = list(throughput_data.values())
            bars = ax3.bar(names, vals, color=self.COLORS['primary'],
                          edgecolor='white', linewidth=1.5, width=0.5)
            for bar, v in zip(bars, vals):
                ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f'{v:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
            ax3.set_ylabel('Events / Second', fontsize=11)
            ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
        else:
            self._empty_panel(ax3, 'No throughput data')
        ax3.set_title('Processing Throughput', fontsize=13, fontweight='bold', pad=10)

        # -- 1d. Memory Usage --
        ax4 = fig.add_subplot(gs[1, 0])
        mem_data = {}
        if parsing_stats:
            mem = self._safe_float(parsing_stats.get('max_memory_mb', 0))
            size = self._safe_float(parsing_stats.get('total_size_mb', 0))
            mem_data['Peak Memory'] = mem
            mem_data['Data Processed'] = size

        if any(v > 0 for v in mem_data.values()):
            names = list(mem_data.keys())
            vals = list(mem_data.values())
            bar_colors = [self.COLORS['secondary'], self.COLORS['info']]
            bars = ax4.bar(names, vals, color=bar_colors, edgecolor='white',
                          linewidth=1.5, width=0.5)
            for bar, v in zip(bars, vals):
                ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f'{v:.1f} MB', ha='center', va='bottom', fontweight='bold', fontsize=10)
            ax4.set_ylabel('MB', fontsize=11)
        else:
            self._empty_panel(ax4, 'No memory data')
        ax4.set_title('Resource Usage', fontsize=13, fontweight='bold', pad=10)

        # -- 1e. File Processing Timeline --
        ax5 = fig.add_subplot(gs[1, 1])
        if parsing_stats:
            total_files = int(parsing_stats.get('total_files', 0))
            succ_files = int(parsing_stats.get('successful_files', 0))
            fail_files = int(parsing_stats.get('failed_files', 0))
            timeouts = int(parsing_stats.get('timeout_count', 0))
            categories = ['Total', 'Success', 'Failed', 'Timeouts']
            values = [total_files, succ_files, fail_files, timeouts]
            bar_colors = [self.COLORS['primary'], self.COLORS['success'],
                         self.COLORS['failure'], self.COLORS['accent']]
            bars = ax5.bar(categories, values, color=bar_colors, edgecolor='white',
                          linewidth=1.5, width=0.6)
            for bar, v in zip(bars, values):
                if v > 0:
                    ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                             str(v), ha='center', va='bottom', fontweight='bold', fontsize=12)
        else:
            self._empty_panel(ax5, 'No file data')
        ax5.set_title('File Processing Breakdown', fontsize=13, fontweight='bold', pad=10)

        # -- 1f. Error Summary --
        ax6 = fig.add_subplot(gs[1, 2])
        error_types = {}
        if parsing_stats:
            error_types = parsing_stats.get('error_types', {})

        if error_types:
            errors = list(error_types.keys())[:5]
            error_counts = [error_types[e] for e in errors]
            # Truncate long error names
            errors = [e[:30] + '...' if len(e) > 30 else e for e in errors]
            bars = ax6.barh(errors, error_counts, color=self.COLORS['failure'],
                           edgecolor='white', linewidth=1.5)
            ax6.set_xlabel('Count', fontsize=11)
        else:
            ax6.text(0.5, 0.5, 'No errors detected',
                     ha='center', va='center', transform=ax6.transAxes,
                     fontsize=16, fontweight='bold', color=self.COLORS['success'])
            ax6.text(0.5, 0.35, 'All files processed successfully',
                     ha='center', va='center', transform=ax6.transAxes,
                     fontsize=11, color='#7f8c8d')
        ax6.set_title('Error Summary', fontsize=13, fontweight='bold', pad=10)

        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        self.logger.info(f"Saved pipeline performance plot to: {output_path}")
        return output_path

    # =========================================================================
    # PLOT 2: Parsed Data Overview
    # =========================================================================
    def plot_parsed_data(
        self,
        parsing_stats: dict,
        particle_stats: Optional[dict] = None,
        save_name: str = "02_parsed_data.png",
    ) -> Optional[Path]:
        """
        Parsed data overview:
        - Total events & chunks (big number cards)
        - Events per file distribution
        - Particle type breakdown (stacked bar)
        - Particle count per-event distributions
        - Chunk size distribution
        - Data volume summary
        """
        self.logger.info("Generating parsed data overview plot...")

        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('Parsed Data Overview', fontsize=20, fontweight='bold',
                     color=self.COLORS['text_dark'], y=0.98)

        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

        # -- 2a. Big Number Cards: Events & Chunks --
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')
        total_events = int(parsing_stats.get('total_events', 0))
        total_chunks = int(parsing_stats.get('total_chunks', 0))
        avg_per_file = self._safe_float(parsing_stats.get('average_events_per_file', 0))

        # Events card
        ax1.text(0.5, 0.82, 'TOTAL EVENTS', ha='center', va='center',
                 transform=ax1.transAxes, fontsize=12, fontweight='bold',
                 color='#7f8c8d')
        ax1.text(0.5, 0.65, f'{total_events:,}', ha='center', va='center',
                 transform=ax1.transAxes, fontsize=36, fontweight='bold',
                 color=self.COLORS['primary'])

        # Chunks card
        ax1.text(0.5, 0.42, 'TOTAL CHUNKS', ha='center', va='center',
                 transform=ax1.transAxes, fontsize=12, fontweight='bold',
                 color='#7f8c8d')
        ax1.text(0.5, 0.25, f'{total_chunks:,}', ha='center', va='center',
                 transform=ax1.transAxes, fontsize=36, fontweight='bold',
                 color=self.COLORS['secondary'])

        # Avg per file
        ax1.text(0.5, 0.08, f'Avg events/file: {avg_per_file:,.0f}', ha='center',
                 va='center', transform=ax1.transAxes, fontsize=11, color='#7f8c8d')
        ax1.set_title('Event Summary', fontsize=13, fontweight='bold', pad=10)

        # -- 2b. Events & Chunks Bar --
        ax2 = fig.add_subplot(gs[0, 1])
        total_files = int(parsing_stats.get('total_files', 0))
        metrics = {
            'Files\nProcessed': total_files,
            'Total\nEvents': total_events,
            'Total\nChunks': total_chunks,
        }
        bar_colors = [self.COLORS['info'], self.COLORS['primary'], self.COLORS['secondary']]
        bars = ax2.bar(range(len(metrics)), list(metrics.values()),
                      color=bar_colors, edgecolor='white', linewidth=1.5, width=0.6)
        ax2.set_xticks(range(len(metrics)))
        ax2.set_xticklabels(metrics.keys(), fontsize=10)
        for bar, v in zip(bars, metrics.values()):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{v:,}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        ax2.set_ylabel('Count', fontsize=11)
        ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
        ax2.set_title('Processing Scale', fontsize=13, fontweight='bold', pad=10)

        # -- 2c. Data Volume --
        ax3 = fig.add_subplot(gs[0, 2])
        size_mb = self._safe_float(parsing_stats.get('total_size_mb', 0))
        time_sec = self._safe_float(parsing_stats.get('total_time_sec', 0))
        ax3.axis('off')
        ax3.text(0.5, 0.75, 'DATA VOLUME', ha='center', va='center',
                 transform=ax3.transAxes, fontsize=12, fontweight='bold', color='#7f8c8d')
        ax3.text(0.5, 0.55, f'{size_mb:.1f} MB', ha='center', va='center',
                 transform=ax3.transAxes, fontsize=28, fontweight='bold',
                 color=self.COLORS['accent'])
        ax3.text(0.5, 0.35, 'PROCESSING TIME', ha='center', va='center',
                 transform=ax3.transAxes, fontsize=12, fontweight='bold', color='#7f8c8d')
        ax3.text(0.5, 0.15, f'{time_sec:.1f} seconds', ha='center', va='center',
                 transform=ax3.transAxes, fontsize=22, fontweight='bold',
                 color=self.COLORS['info'])
        ax3.set_title('Volume & Timing', fontsize=13, fontweight='bold', pad=10)

        # -- 2d. Particle Type Breakdown --
        ax4 = fig.add_subplot(gs[1, 0])
        if particle_stats and 'particle_counts' in particle_stats:
            p_counts = particle_stats['particle_counts']
            if p_counts:
                particles = list(p_counts.keys())
                counts = [int(v) for v in p_counts.values()]
                p_colors = [self.COLORS.get(p.lower(), self.COLORS['primary']) for p in particles]
                bars = ax4.barh(particles, counts, color=p_colors, edgecolor='white',
                               linewidth=1.5, height=0.6)
                for bar, v in zip(bars, counts):
                    ax4.text(bar.get_width() + max(counts) * 0.02,
                             bar.get_y() + bar.get_height() / 2,
                             f'{v:,}', va='center', fontweight='bold', fontsize=10)
                ax4.set_xlabel('Total Particle Count', fontsize=11)
                ax4.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
            else:
                self._empty_panel(ax4, 'No particle data')
        else:
            self._empty_panel(ax4, 'Particle stats not available')
        ax4.set_title('Particle Type Distribution', fontsize=13, fontweight='bold', pad=10)

        # -- 2e. Particle Count Per-Event Distribution --
        ax5 = fig.add_subplot(gs[1, 1])
        if particle_stats and 'distributions' in particle_stats:
            distributions = particle_stats['distributions']
            if distributions:
                for idx, (ptype, dist) in enumerate(distributions.items()):
                    if dist:
                        arr = np.array(dist)
                        color = self.COLORS.get(ptype.lower(), self.COLORS['primary'])
                        ax5.hist(arr, bins=min(30, max(arr) - min(arr) + 1),
                                alpha=0.6, label=ptype, color=color, edgecolor='white')
                ax5.set_xlabel('Count per Event', fontsize=11)
                ax5.set_ylabel('Frequency', fontsize=11)
                ax5.legend(fontsize=9, frameon=True, fancybox=True)
            else:
                self._empty_panel(ax5, 'No distribution data')
        else:
            self._empty_panel(ax5, 'Per-event distributions\nnot available')
        ax5.set_title('Particles Per Event', fontsize=13, fontweight='bold', pad=10)

        # -- 2f. Events per File --
        ax6 = fig.add_subplot(gs[1, 2])
        if particle_stats and 'events_per_file' in particle_stats:
            epf = particle_stats['events_per_file']
            if epf:
                ax6.hist(epf, bins=min(20, len(epf)), color=self.COLORS['primary'],
                         alpha=0.7, edgecolor='white')
                ax6.axvline(np.mean(epf), color=self.COLORS['failure'],
                           linestyle='--', linewidth=2, label=f'Mean: {np.mean(epf):,.0f}')
                ax6.set_xlabel('Events per File', fontsize=11)
                ax6.set_ylabel('Frequency', fontsize=11)
                ax6.legend(fontsize=10, frameon=True)
            else:
                self._empty_panel(ax6, 'No per-file data')
        else:
            self._empty_panel(ax6, 'Events per file\nnot available')
        ax6.set_title('Events Per File Distribution', fontsize=13, fontweight='bold', pad=10)

        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        self.logger.info(f"Saved parsed data plot to: {output_path}")
        return output_path

    # =========================================================================
    # PLOT 3: Invariant Mass Summary
    # =========================================================================
    def plot_invariant_mass_summary(
        self,
        mass_calc_stats: dict,
        save_name: str = "03_invariant_mass.png",
    ) -> Optional[Path]:
        """
        Invariant mass calculation summary:
        - Total final states created
        - Combination counts per object type
        - Object contribution pie chart
        - Combination size distribution
        - Final state event counts
        - Processing efficiency
        """
        self.logger.info("Generating invariant mass summary plot...")

        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('Invariant Mass Calculation Summary', fontsize=20, fontweight='bold',
                     color=self.COLORS['text_dark'], y=0.98)

        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

        # -- 3a. Big Numbers: Final States & Combinations --
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')
        total_fs = int(mass_calc_stats.get('total_final_states', 0))
        total_combos = int(mass_calc_stats.get('total_combinations', 0))
        total_events = int(mass_calc_stats.get('total_events_processed', 0))

        ax1.text(0.5, 0.85, 'FINAL STATES', ha='center', va='center',
                 transform=ax1.transAxes, fontsize=12, fontweight='bold', color='#7f8c8d')
        ax1.text(0.5, 0.68, f'{total_fs:,}', ha='center', va='center',
                 transform=ax1.transAxes, fontsize=36, fontweight='bold',
                 color=self.COLORS['primary'])
        ax1.text(0.5, 0.48, 'COMBINATIONS', ha='center', va='center',
                 transform=ax1.transAxes, fontsize=12, fontweight='bold', color='#7f8c8d')
        ax1.text(0.5, 0.31, f'{total_combos:,}', ha='center', va='center',
                 transform=ax1.transAxes, fontsize=36, fontweight='bold',
                 color=self.COLORS['secondary'])
        ax1.text(0.5, 0.12, f'Events processed: {total_events:,}', ha='center',
                 va='center', transform=ax1.transAxes, fontsize=11, color='#7f8c8d')
        ax1.set_title('Calculation Summary', fontsize=13, fontweight='bold', pad=10)

        # -- 3b. Combinations Per Object Type --
        ax2 = fig.add_subplot(gs[0, 1])
        combos_per_object = mass_calc_stats.get('combinations_per_object', {})
        if combos_per_object:
            objects = list(combos_per_object.keys())
            counts = [int(v) for v in combos_per_object.values()]
            obj_colors = [self.COLORS.get(o.lower(), self.COLORS['primary']) for o in objects]
            bars = ax2.bar(objects, counts, color=obj_colors, edgecolor='white',
                          linewidth=1.5, width=0.6)
            for bar, v in zip(bars, counts):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f'{v:,}', ha='center', va='bottom', fontweight='bold', fontsize=11)
            ax2.set_ylabel('Combinations', fontsize=11)
        else:
            self._empty_panel(ax2, 'No object data')
        ax2.set_title('Combinations Per Object', fontsize=13, fontweight='bold', pad=10)

        # -- 3c. Object Contribution Pie --
        ax3 = fig.add_subplot(gs[0, 2])
        if combos_per_object and sum(combos_per_object.values()) > 0:
            objects = list(combos_per_object.keys())
            counts = [int(v) for v in combos_per_object.values()]
            obj_colors = [self.COLORS.get(o.lower(), self.COLORS['primary']) for o in objects]
            wedges, texts, autotexts = ax3.pie(
                counts, labels=objects, colors=obj_colors,
                autopct='%1.1f%%', startangle=90,
                wedgeprops=dict(edgecolor='white', linewidth=2))
            for at in autotexts:
                at.set_fontweight('bold')
        else:
            self._empty_panel(ax3, 'No combination data')
        ax3.set_title('Object Contribution', fontsize=13, fontweight='bold', pad=10)

        # -- 3d. Combination Size Distribution --
        ax4 = fig.add_subplot(gs[1, 0])
        combo_sizes = mass_calc_stats.get('combination_size_distribution', {})
        if combo_sizes:
            sizes = sorted(combo_sizes.keys())
            counts = [combo_sizes[s] for s in sizes]
            bars = ax4.bar([f'{s}-body' for s in sizes], counts,
                          color=self.COLORS['accent'], edgecolor='white', linewidth=1.5)
            for bar, v in zip(bars, counts):
                ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f'{v:,}', ha='center', va='bottom', fontweight='bold', fontsize=11)
            ax4.set_ylabel('Number of Combinations', fontsize=11)
        else:
            self._empty_panel(ax4, 'No size distribution data')
        ax4.set_title('Combination Size Distribution', fontsize=13, fontweight='bold', pad=10)

        # -- 3e. Final State Event Counts (top N) --
        ax5 = fig.add_subplot(gs[1, 1])
        fs_events = mass_calc_stats.get('events_per_final_state', {})
        if fs_events:
            # Sort and take top 10
            sorted_fs = sorted(fs_events.items(), key=lambda x: x[1], reverse=True)[:10]
            names = [n[:20] for n, _ in sorted_fs]
            counts = [c for _, c in sorted_fs]
            bars = ax5.barh(names, counts, color=self.COLORS['info'],
                           edgecolor='white', linewidth=1.5)
            for bar, v in zip(bars, counts):
                ax5.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                         f' {v:,}', va='center', fontweight='bold', fontsize=9)
            ax5.set_xlabel('Events', fontsize=11)
            ax5.invert_yaxis()
        else:
            self._empty_panel(ax5, 'No final state event data')
        ax5.set_title('Top Final States by Events', fontsize=13, fontweight='bold', pad=10)

        # -- 3f. Processing Efficiency --
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        proc_time = self._safe_float(mass_calc_stats.get('total_time_sec', 0))
        fs_per_sec = total_fs / proc_time if proc_time > 0 else 0
        combos_per_sec = total_combos / proc_time if proc_time > 0 else 0

        ax6.text(0.5, 0.80, 'PROCESSING RATE', ha='center', va='center',
                 transform=ax6.transAxes, fontsize=12, fontweight='bold', color='#7f8c8d')
        ax6.text(0.5, 0.62, f'{combos_per_sec:,.1f}', ha='center', va='center',
                 transform=ax6.transAxes, fontsize=30, fontweight='bold',
                 color=self.COLORS['primary'])
        ax6.text(0.5, 0.48, 'combinations / second', ha='center', va='center',
                 transform=ax6.transAxes, fontsize=11, color='#7f8c8d')

        ax6.text(0.5, 0.28, f'{fs_per_sec:,.1f} final states/sec', ha='center',
                 va='center', transform=ax6.transAxes, fontsize=13,
                 fontweight='bold', color=self.COLORS['secondary'])
        ax6.text(0.5, 0.12, f'Total time: {proc_time:.1f}s', ha='center',
                 va='center', transform=ax6.transAxes, fontsize=11, color='#7f8c8d')
        ax6.set_title('Efficiency Metrics', fontsize=13, fontweight='bold', pad=10)

        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        self.logger.info(f"Saved invariant mass summary plot to: {output_path}")
        return output_path

    # =========================================================================
    # PLOT 4: Histogram Metadata
    # =========================================================================
    def plot_histogram_metadata(
        self,
        histogram_stats: dict,
        save_name: str = "04_histogram_metadata.png",
    ) -> Optional[Path]:
        """
        Histogram creation metadata:
        - Total histograms created
        - Main vs outlier breakdown
        - Peaks detected & cut
        - Histogram entries distribution
        - Bin width & range summary
        - Final state coverage
        """
        self.logger.info("Generating histogram metadata plot...")

        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('Histogram Creation Metadata', fontsize=20, fontweight='bold',
                     color=self.COLORS['text_dark'], y=0.98)

        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

        # -- 4a. Big Numbers: Histograms Created --
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')
        total_hists = int(histogram_stats.get('total_histograms', 0))
        main_hists = int(histogram_stats.get('main_histograms', 0))
        outlier_hists = int(histogram_stats.get('outlier_histograms', 0))

        ax1.text(0.5, 0.85, 'TOTAL HISTOGRAMS', ha='center', va='center',
                 transform=ax1.transAxes, fontsize=12, fontweight='bold', color='#7f8c8d')
        ax1.text(0.5, 0.65, f'{total_hists:,}', ha='center', va='center',
                 transform=ax1.transAxes, fontsize=40, fontweight='bold',
                 color=self.COLORS['primary'])
        ax1.text(0.3, 0.35, f'Main: {main_hists:,}', ha='center', va='center',
                 transform=ax1.transAxes, fontsize=14, fontweight='bold',
                 color=self.COLORS['success'])
        ax1.text(0.7, 0.35, f'Outlier: {outlier_hists:,}', ha='center', va='center',
                 transform=ax1.transAxes, fontsize=14, fontweight='bold',
                 color=self.COLORS['failure'])
        ax1.set_title('Histogram Count', fontsize=13, fontweight='bold', pad=10)

        # -- 4b. Main vs Outlier Donut --
        ax2 = fig.add_subplot(gs[0, 1])
        if main_hists > 0 or outlier_hists > 0:
            wedges, _, autotexts = ax2.pie(
                [main_hists, outlier_hists],
                labels=None,
                colors=[self.COLORS['success'], self.COLORS['failure']],
                autopct='%1.1f%%', startangle=90,
                pctdistance=0.78,
                wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2),
            )
            for at in autotexts:
                at.set_fontweight('bold')
                at.set_fontsize(11)
            pct = main_hists / (main_hists + outlier_hists) * 100
            ax2.text(0, 0, f'{pct:.0f}%\nmain', ha='center', va='center',
                     fontsize=18, fontweight='bold', color=self.COLORS['text_dark'])
            ax2.legend(['Main', 'Outlier'], loc='lower center',
                       fontsize=10, ncol=2, frameon=False)
        else:
            self._empty_panel(ax2, 'No histogram data')
        ax2.set_title('Main vs Outlier', fontsize=13, fontweight='bold', pad=10)

        # -- 4c. Peaks Detected & Cut --
        ax3 = fig.add_subplot(gs[0, 2])
        peaks_detected = int(histogram_stats.get('peaks_detected', 0))
        peaks_cut = int(histogram_stats.get('peaks_cut', 0))
        hists_with_peaks = int(histogram_stats.get('histograms_with_peaks_cut', 0))

        peak_data = {
            'Peaks\nDetected': peaks_detected,
            'Peaks\nCut': peaks_cut,
            'Histograms\nw/ Cuts': hists_with_peaks,
        }
        if any(v > 0 for v in peak_data.values()):
            bar_colors = [self.COLORS['accent'], self.COLORS['failure'], self.COLORS['secondary']]
            bars = ax3.bar(range(len(peak_data)), list(peak_data.values()),
                          color=bar_colors, edgecolor='white', linewidth=1.5, width=0.6)
            ax3.set_xticks(range(len(peak_data)))
            ax3.set_xticklabels(peak_data.keys(), fontsize=10)
            for bar, v in zip(bars, peak_data.values()):
                ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f'{v:,}', ha='center', va='bottom', fontweight='bold', fontsize=12)
            ax3.set_ylabel('Count', fontsize=11)
        else:
            self._empty_panel(ax3, 'No peak data available\n(Run post-processing first)')
        ax3.set_title('Peak Detection & Removal', fontsize=13, fontweight='bold', pad=10)

        # -- 4d. Histogram Entries Distribution --
        ax4 = fig.add_subplot(gs[1, 0])
        entries_dist = histogram_stats.get('entries_per_histogram', [])
        if entries_dist:
            ax4.hist(entries_dist, bins=min(30, len(entries_dist)),
                     color=self.COLORS['primary'], alpha=0.7, edgecolor='white')
            ax4.axvline(np.mean(entries_dist), color=self.COLORS['failure'],
                       linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(entries_dist):,.0f}')
            ax4.axvline(np.median(entries_dist), color=self.COLORS['accent'],
                       linestyle=':', linewidth=2,
                       label=f'Median: {np.median(entries_dist):,.0f}')
            ax4.set_xlabel('Entries per Histogram', fontsize=11)
            ax4.set_ylabel('Frequency', fontsize=11)
            ax4.legend(fontsize=10, frameon=True)
        else:
            self._empty_panel(ax4, 'No entry distribution data')
        ax4.set_title('Histogram Entries Distribution', fontsize=13, fontweight='bold', pad=10)

        # -- 4e. Final State Coverage --
        ax5 = fig.add_subplot(gs[1, 1])
        fs_histogram_counts = histogram_stats.get('histograms_per_final_state', {})
        if fs_histogram_counts:
            sorted_fs = sorted(fs_histogram_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            names = [n[:20] for n, _ in sorted_fs]
            counts = [c for _, c in sorted_fs]
            bars = ax5.barh(names, counts, color=self.COLORS['info'],
                           edgecolor='white', linewidth=1.5)
            for bar, v in zip(bars, counts):
                ax5.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                         f' {v:,}', va='center', fontweight='bold', fontsize=9)
            ax5.set_xlabel('Histograms', fontsize=11)
            ax5.invert_yaxis()
        else:
            self._empty_panel(ax5, 'No final state coverage data')
        ax5.set_title('Histograms Per Final State', fontsize=13, fontweight='bold', pad=10)

        # -- 4f. Configuration Summary --
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        bin_width = histogram_stats.get('bin_width_gev', 'N/A')
        exclude_outliers = histogram_stats.get('exclude_outliers', 'N/A')
        bumpnet = histogram_stats.get('use_bumpnet_naming', 'N/A')
        output_file = histogram_stats.get('output_filename', 'N/A')

        config_lines = [
            ('Bin Width', f'{bin_width} GeV'),
            ('Exclude Outliers', str(exclude_outliers)),
            ('BumpNet Naming', str(bumpnet)),
            ('Output File', str(output_file)),
            ('Total Created', f'{total_hists:,}'),
        ]

        y_start = 0.85
        for i, (label, value) in enumerate(config_lines):
            y = y_start - i * 0.15
            ax6.text(0.1, y, label + ':', ha='left', va='center',
                     transform=ax6.transAxes, fontsize=12, fontweight='bold',
                     color='#7f8c8d')
            ax6.text(0.9, y, value, ha='right', va='center',
                     transform=ax6.transAxes, fontsize=12, fontweight='bold',
                     color=self.COLORS['text_dark'])
        ax6.set_title('Configuration', fontsize=13, fontweight='bold', pad=10)

        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        self.logger.info(f"Saved histogram metadata plot to: {output_path}")
        return output_path

    # =========================================================================
    # Orchestrator
    # =========================================================================
    def create_all_plots(self, pipeline_stats: dict) -> List[Path]:
        """
        Create all applicable statistical plots based on available data.

        Args:
            pipeline_stats: Dict with keys: 'parsing', 'particles', 'mass_calc',
                            'post_processing', 'histograms'

        Returns:
            List of paths to created plots
        """
        self.logger.info("Creating all statistical plots...")
        created_plots = []

        parsing_stats = pipeline_stats.get('parsing')
        particle_stats = pipeline_stats.get('particles')
        mass_calc_stats = pipeline_stats.get('mass_calc')
        post_proc_stats = pipeline_stats.get('post_processing')
        hist_stats = pipeline_stats.get('histograms')

        # 1. Pipeline Performance (always generated if any stats exist)
        if parsing_stats or mass_calc_stats or post_proc_stats or hist_stats:
            try:
                path = self.plot_pipeline_performance(
                    parsing_stats=parsing_stats,
                    mass_calc_stats=mass_calc_stats,
                    post_proc_stats=post_proc_stats,
                    hist_stats=hist_stats,
                )
                if path:
                    created_plots.append(path)
            except Exception as e:
                self.logger.warning(f"Failed to create pipeline performance plot: {e}")

        # 2. Parsed Data Overview (needs parsing stats)
        if parsing_stats:
            try:
                path = self.plot_parsed_data(
                    parsing_stats=parsing_stats,
                    particle_stats=particle_stats,
                )
                if path:
                    created_plots.append(path)
            except Exception as e:
                self.logger.warning(f"Failed to create parsed data plot: {e}")

        # 3. Invariant Mass Summary (needs mass calc stats)
        if mass_calc_stats:
            try:
                path = self.plot_invariant_mass_summary(mass_calc_stats=mass_calc_stats)
                if path:
                    created_plots.append(path)
            except Exception as e:
                self.logger.warning(f"Failed to create invariant mass plot: {e}")

        # 4. Histogram Metadata (needs histogram stats)
        if hist_stats:
            try:
                path = self.plot_histogram_metadata(histogram_stats=hist_stats)
                if path:
                    created_plots.append(path)
            except Exception as e:
                self.logger.warning(f"Failed to create histogram metadata plot: {e}")

        self.logger.info(f"Created {len(created_plots)} statistical plots")
        return created_plots

    # =========================================================================
    # Helpers
    # =========================================================================
    @staticmethod
    def _safe_float(val) -> float:
        """Safely convert a value to float, handling strings like '12.5' or '100.0%'."""
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            try:
                return float(val.rstrip('%'))
            except ValueError:
                return 0.0
        return 0.0

    @staticmethod
    def _empty_panel(ax, message: str):
        """Render an empty panel with a placeholder message."""
        ax.text(0.5, 0.5, message, ha='center', va='center',
                transform=ax.transAxes, fontsize=13, color='#95a5a6',
                style='italic')
        ax.set_xticks([])
        ax.set_yticks([])
