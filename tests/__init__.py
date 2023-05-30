
import contextlib
import io


def run_main(main, args):
    f = io.StringIO()

    try:
        with contextlib.redirect_stdout(f):
            try:
                main(args)
            except SystemExit:
                pass
    except:
        print(f.getvalue())
        raise

    return f.getvalue()


def capture_stdout(func, *args, **kwargs):
    f = io.StringIO()

    try:
        with contextlib.redirect_stdout(f):
            func(*args, **kwargs)
    except:
        print(f.getvalue())
        raise

    return f.getvalue()
