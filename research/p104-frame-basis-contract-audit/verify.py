"""Two finite frame counterexamples, not a multiplication or reset proof.
Prepared for SYLforge with substantial OpenAI assistance. Apache-2.0.
"""
import hashlib, json, sys, types
from pathlib import Path
P = Path(__file__).resolve().parent

def need(ok, text):
    if not ok:
        raise ValueError(text)

def span(vectors):
    s = {0}
    for v in vectors:
        s |= {a ^ v for a in s}
    return s

def main():
    need(not sys.flags.optimize, 'Run without -O or -OO')
    source = (P/'vendor/native_verify.py').read_bytes()
    need(hashlib.sha256(source).hexdigest() == '3049471905b47b385c88378a1a3be7a18b8858aadd035b8a6da2f4a03b470f38', 'Interpreter source hash')
    code = source.decode()
    compat = not hasattr(int, 'bit_count')
    if compat:
        need(code.count('value.bit_count()') == 1, 'Exact compatibility substitution')
        code = code.replace('value.bit_count()', 'bin(value).count("1")')
    m = types.ModuleType('pinned_frame_interpreter')
    exec(compile(code, 'vendor/native_verify.py', 'exec'), m.__dict__)
    cases = [
        (1090, [265,194,164,75], [45,75,137,265], 0),
        (1091, [11,21,51,469], [11,21,38,448], 16),
    ]
    rows = []
    for label, old, changed, column in cases:
        need(old != changed and span(old) == span(changed), 'Same span but changed basis')
        a, b = m.frame(tuple(old),9), m.frame(tuple(changed),9)
        m.equal(a, a, 'Literal original positive control')
        need(m.tableau(a) != m.tableau(b), 'Different signed Pauli action')
        need(m.tableau(m.adjoint(a)) != m.tableau(m.adjoint(b)), 'Adjoints differ too')
        need(m.state(a,column) != m.state(b,column), 'Exact witness column differs')
        rejected = False
        try:
            m.equal(a,b,'Changed frame')
        except ValueError as e:
            rejected = 'signed Pauli action' in str(e)
        need(rejected, 'Reject span-only acceptance')
        fixed = m.frame(tuple(old),9)
        m.equal(a,fixed,'Restoring literal basis restores operator')
        phase_bad = a + (('phase',1),)
        rejected_phase = False
        try:
            m.equal(a,phase_bad,'Wrong global phase')
        except ValueError:
            rejected_phase = True
        need(rejected_phase, 'Reject global-phase mutation')
        rows.append(dict(frame=label,original_basis=old,changed_basis=changed,
                         span_equal=True,operator_equal=False,adjoint_equal=False,
                         witness_column=column,restored_operator_equal=True,
                         wrong_phase_rejected=True))
    print(json.dumps(dict(status='PASS_TWO_FRAME_COUNTEREXAMPLES_ONLY',
        python=sys.version.split()[0],python39_parity_compatibility=compat,
        scope='Finite frame regression only; no full word, invoice, erasure, or supplier admission',
        frames=rows),indent=2))

if __name__ == '__main__':
    main()
