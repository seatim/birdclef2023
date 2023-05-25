
import contextlib
import io


def run_main(main, args):
    f = io.StringIO()

    with contextlib.redirect_stdout(f):
        try:
            main(args)
        except SystemExit:
            pass

    return f.getvalue()
