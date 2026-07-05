#!/usr/bin/env bash
# ============================================================================
# cleanup.sh  -  Repo-Aufraeumen MasterArbeit (Stand 2026-07-05, v2)
#
# v2-Aenderungen gegenueber v1 (Review):
#   - model_compare: answers werden vor dem Loeschen per cmp gegen die
#     kanonische Version geprueft; die SLURM-.out-Logs (Provenienz der
#     kanonischen Laeufe) werden nach results/phase3/metadata/slurm_logs/
#     konsolidiert statt nur im Backup zu landen.
#   - slurm/pilot_r1_14b.sbatch bleibt (billiger Voraustest fuer die
#     geplante Denkbudget-Kurve).
#   - Randfall abgefangen: nur noch __pycache__ uebrig -> sauberer Lauf
#     statt tar-Fehler.
#
# Ablauf, in genau dieser Reihenfolge:
#   1. KONSOLIDIEREN: die kanonischen Phase-3-Artefakte (answers_qwen14b/
#      qwen32b/r1_32b.jsonl, metadata/) landen direkt in results/phase3/.
#      Quelle ist das FIXED-Bundle (entpackter Ordner, sonst das Zip).
#      Jede Datei wird per sha256 gegen die in der Claude-Session aus dem
#      Bundle nachgerechneten Referenzwerte geprueft; prompts.jsonl und
#      pilot_cases.csv im Repo ebenso. Bei JEDER Abweichung: Abbruch,
#      bevor irgendetwas geloescht wird.
#   2. BACKUP: alle Loeschkandidaten (ausser src/__pycache__) wandern in
#      ein tar.gz NEBEN dem Repo:
#          ../MasterArbeit_cleanup_backup_<stamp>.tar.gz
#      Wiederherstellen einzelner Dinge:
#          tar -xzf ../MasterArbeit_cleanup_backup_<stamp>.tar.gz -C . <pfad>
#   3. LOESCHEN: erst nach erfolgreichem Backup.
#
# Aufruf im Projektroot (~/Dokumente/MasterArbeit):
#   bash cleanup.sh --dry-run    # nur pruefen und anzeigen, nichts aendern
#   bash cleanup.sh              # ausfuehren
#
# Ein zweiter Lauf ist ungefaehrlich: bereits konsolidierte Ziele werden
# nur noch verifiziert, fehlende Kandidaten uebersprungen.
# ============================================================================
set -euo pipefail

DRY=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY=1; fi

# ---- 0. Kontext pruefen ----------------------------------------------------
if [[ ! -f src/walks.py || ! -d results/phase3 ]]; then
    echo "ABBRUCH: nicht im Projektroot (src/walks.py oder results/phase3 fehlt)." >&2
    exit 1
fi

P3="results/phase3"
BDIR="$P3/claude_minimal_results_bundle_FIXED_20260703_100929"
BZIP="$BDIR.zip"
STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="../MasterArbeit_cleanup_backup_${STAMP}.tar.gz"

# Referenzwerte, in der Claude-Sandbox direkt aus dem FIXED-Bundle berechnet
# (dieselben Dateien, aus denen die HANDOFF-Tabellen exakt reproduziert wurden)
SHA_Q14="30b1d5890e12971c6df9e6d071647f0a279443214c472cda2d3cbec645522556"
SHA_Q32="eaeebf8eb1986cf32092ec734000fed886eb8bc0ebdffae7ea88afdfbfbda380"
SHA_R1="3ef399931f804eb3e6ac29e0aafa6cd1d7ce36223590fed30c6d784d52a4b87d"
SHA_PROMPTS="517f736fc7f2d8520ad0baac79c8c84893d5a8437e41a2c08b2c0e5e6781c3fe"
SHA_CASES="65c6693969fdf6b89376e764c2de6cc2f02183e08f4d7cc88a8af05f7abd60bc"

fail() { echo "ABBRUCH: $*" >&2; echo "Nichts wurde geloescht." >&2; exit 1; }

check_sha() {  # $1 = Datei, $2 = erwarteter Hash, $3 = Label
    if [[ ! -f "$1" ]]; then fail "$3: Datei fehlt ($1)"; fi
    local got
    got="$(sha256sum "$1" | cut -d' ' -f1)"
    if [[ "$got" != "$2" ]]; then
        fail "$3: Checksumme weicht ab ($1)
  erwartet: $2
  erhalten: $got
Bitte nicht weitermachen, sondern melden."
    fi
    echo "  ok  $3"
}

echo "== 1/3 Verifikation + Konsolidierung =="

# Der Repo-Stand muss dem verifizierten Stand entsprechen
check_sha "$P3/prompts.jsonl"   "$SHA_PROMPTS" "prompts.jsonl (v2, Repo)"
check_sha "$P3/pilot_cases.csv" "$SHA_CASES"   "pilot_cases.csv (Repo)"

# Quelle fuer answers + metadata finden: entpackter Ordner, sonst Zip
SRC=""
TMP=""
if [[ -f "$BDIR/results/answers_qwen14b.jsonl" ]]; then
    SRC="$BDIR"
elif [[ -f "$BZIP" ]]; then
    command -v unzip >/dev/null 2>&1 || fail "unzip nicht installiert"
    TMP="$(mktemp -d)"
    trap '[[ -n "$TMP" ]] && rm -rf "$TMP"' EXIT
    unzip -q "$BZIP" -d "$TMP"
    SRC="$TMP"
fi

TARGETS=(answers_qwen14b.jsonl answers_qwen32b.jsonl answers_r1_32b.jsonl)
SHAS=("$SHA_Q14" "$SHA_Q32" "$SHA_R1")

if [[ -n "$SRC" ]]; then
    for i in 0 1 2; do
        f="${TARGETS[$i]}"
        check_sha "$SRC/results/$f" "${SHAS[$i]}" "$f (Bundle)"
        if [[ -f "$P3/$f" ]]; then
            cmp -s "$SRC/results/$f" "$P3/$f" \
                || fail "$P3/$f existiert bereits und weicht vom Bundle ab"
        elif [[ "$DRY" -eq 0 ]]; then
            cp "$SRC/results/$f" "$P3/$f"
        fi
    done
    if [[ ! -d "$P3/metadata" && "$DRY" -eq 0 ]]; then
        cp -r "$SRC/metadata" "$P3/metadata"
    fi
    if [[ "$DRY" -eq 1 ]]; then
        echo "  (dry-run) answers x3 und metadata/ wuerden nach $P3/ kopiert"
    else
        echo "  kanonische Artefakte liegen jetzt in $P3/ (answers x3, metadata/)"
    fi
else
    # kein Bundle mehr da (z. B. zweiter Lauf): Ziele muessen korrekt vorliegen
    for i in 0 1 2; do
        check_sha "$P3/${TARGETS[$i]}" "${SHAS[$i]}" "${TARGETS[$i]} (Repo)"
    done
    [[ -d "$P3/metadata" ]] || fail "$P3/metadata fehlt und kein Bundle als Quelle da"
    echo "  Bundle nicht mehr vorhanden, Ziele bereits konsolidiert und verifiziert"
fi

# ---- model_compare absichern, bevor es geloescht wird ----------------------
MC="$P3/model_compare"
if [[ -d "$MC" ]]; then
    for i in 0 1 2; do
        f="${TARGETS[$i]}"
        [[ -f "$MC/$f" ]] || continue
        ref=""
        if [[ -f "$P3/$f" ]]; then ref="$P3/$f"
        elif [[ -n "$SRC" && -f "$SRC/results/$f" ]]; then ref="$SRC/results/$f"
        fi
        [[ -n "$ref" ]] || fail "keine kanonische Referenz fuer model_compare/$f gefunden"
        cmp -s "$MC/$f" "$ref" \
            || fail "model_compare/$f weicht von der kanonischen Version ab. Bitte melden statt loeschen."
        echo "  ok  model_compare/$f == kanonische Version"
    done
    if compgen -G "$MC/*.out" > /dev/null; then
        n_out="$(ls "$MC"/*.out | wc -l)"
        if [[ "$DRY" -eq 1 ]]; then
            echo "  (dry-run) $n_out SLURM-Logs wuerden nach $P3/metadata/slurm_logs/ kopiert"
        else
            mkdir -p "$P3/metadata/slurm_logs"
            cp -f "$MC"/*.out "$P3/metadata/slurm_logs/"
            echo "  $n_out SLURM-Logs konsolidiert: $P3/metadata/slurm_logs/"
        fi
    fi
fi

echo
echo "== 2/3 Backup der Loeschkandidaten =="

CAND=(
    src/__pycache__
    src/make_pilot_cases.py
    src/run_llm_pilot.py
    src/run_llm_pilot_v2.py
    src/score_pilot.py
    src/phase3_prepare.py
    experiments
    docs/STATE_current.md
    data/synthetic/main_run1
    "$P3/answers.jsonl"
    "$P3/answers_qwen14b_v1_backup.jsonl"
    "$P3/prompts_v1_backup.jsonl"
    "$P3/make_pilot_cases.log"
    "$P3/pilot_run_5704629.out"
    "$P3/pilot_smoke_5704597.out"
    "$P3/pilot_score.log"
    "$P3/pilot_mae_table.csv"
    "$P3/pilot_paired.csv"
    "$P3/pilot_band_plot.png"
    "$P3/phase3_pilot_results_for_claude.zip"
    "$P3/claude_minimal_results_bundle_20260703_100812"
    "$P3/claude_minimal_results_bundle_20260703_100812.zip"
    "$BDIR"
    "$BZIP"
    "$P3/model_compare"
    slurm/pilot_run.sbatch
    slurm/pilot_smoke.sbatch
)

EXIST=()
MISS=()
for p in "${CAND[@]}"; do
    if [[ -e "$p" ]]; then EXIST+=("$p"); else MISS+=("$p"); fi
done
if [[ "${#EXIST[@]}" -eq 0 ]]; then
    echo "  keiner der Loeschkandidaten existiert mehr, nichts zu tun."
    exit 0
fi

echo "  zu loeschen (${#EXIST[@]} Eintraege, Groesse):"
du -sh -- "${EXIST[@]}" | sed 's/^/    /'
if [[ "${#MISS[@]}" -gt 0 ]]; then
    echo "  nicht gefunden, uebersprungen:"
    printf '    %s\n' "${MISS[@]}"
fi

TARL=()
for p in "${EXIST[@]}"; do
    if [[ "$p" != "src/__pycache__" ]]; then TARL+=("$p"); fi
done

if [[ "$DRY" -eq 1 ]]; then
    echo
    echo "== DRY-RUN: nichts kopiert, gesichert oder geloescht. =="
    echo "   Backup wuerde nach $BACKUP geschrieben."
    exit 0
fi

if [[ "${#TARL[@]}" -gt 0 ]]; then
    tar -czf "$BACKUP" -- "${TARL[@]}"
    echo "  Backup geschrieben: $BACKUP ($(du -sh -- "$BACKUP" | cut -f1))"
else
    echo "  nur src/__pycache__ vorhanden, kein Backup noetig."
fi

echo
echo "== 3/3 Loeschen =="
rm -rf -- "${EXIST[@]}"
if [[ -d docs && -z "$(ls -A docs)" ]]; then
    rmdir docs
    echo "  docs/ (leer) entfernt"
fi
if [[ -d data/synthetic && -z "$(ls -A data/synthetic)" ]]; then
    rmdir data/synthetic
    echo "  data/synthetic/ (leer) entfernt"
fi
echo "  fertig."

echo
echo "== Ergebnis =="
echo "Repo-Groesse jetzt: $(du -sh . | cut -f1)"
echo
echo "results/phase3/:"
find "$P3" -maxdepth 2 | sort | sed 's/^/  /'
echo
echo "src/:"
ls -1 src | sed 's/^/  /'
echo
echo "slurm/:"
ls -1 slurm | sed 's/^/  /'
echo
if [[ -f "$BACKUP" ]]; then
    echo "Backup liegt neben dem Repo: $BACKUP"
    echo "Nach ein paar Wochen ohne Vermissen: rm '$BACKUP'"
fi
