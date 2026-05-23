"""SQLite wrapper for the collaborative photo album feature."""
import sqlite3
import os

DB_PATH = os.environ.get('DB_PATH', os.path.join(os.path.dirname(__file__), 'album.db'))


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init():
    conn = _connect()
    try:
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS albums (
                id         TEXT PRIMARY KEY,
                name       TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS photos (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                album_id        TEXT    NOT NULL,
                filename        TEXT    NOT NULL,
                style           TEXT    NOT NULL,
                uploaded_by     TEXT    NOT NULL,
                uploaded_at     TEXT    NOT NULL,
                faces_detected  INTEGER NOT NULL DEFAULT 0
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS tags (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                photo_id INTEGER NOT NULL,
                face_id  INTEGER,
                name     TEXT    NOT NULL,
                UNIQUE(photo_id, name)
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                photo_id          INTEGER NOT NULL,
                crop_url          TEXT    NOT NULL,
                embedding         BLOB,
                person_id         INTEGER,
                match_confidence  TEXT
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS people (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                album_id   TEXT    NOT NULL,
                name       TEXT,
                centroid   BLOB    NOT NULL,
                face_count INTEGER NOT NULL DEFAULT 1
            )
        ''')
        conn.commit()
        # Indexes for hot queries
        for idx in [
            'CREATE INDEX IF NOT EXISTS idx_photos_album  ON photos(album_id)',
            'CREATE INDEX IF NOT EXISTS idx_photos_id     ON photos(album_id, id)',
            'CREATE INDEX IF NOT EXISTS idx_faces_photo   ON faces(photo_id)',
            'CREATE INDEX IF NOT EXISTS idx_faces_person  ON faces(person_id)',
            'CREATE INDEX IF NOT EXISTS idx_tags_photo    ON tags(photo_id)',
        ]:
            conn.execute(idx)
        conn.commit()
        # Migrations for existing DBs that predate these columns
        for stmt in [
            'ALTER TABLE photos  ADD COLUMN faces_detected INTEGER NOT NULL DEFAULT 0',
            'ALTER TABLE tags    ADD COLUMN face_id INTEGER',
            'ALTER TABLE faces   ADD COLUMN embedding BLOB',
            'ALTER TABLE faces   ADD COLUMN person_id INTEGER',
            'ALTER TABLE faces   ADD COLUMN match_confidence TEXT',
            'ALTER TABLE people  ADD COLUMN name TEXT',
        ]:
            try:
                conn.execute(stmt)
                conn.commit()
            except sqlite3.OperationalError:
                pass  # column already exists
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Albums
# ---------------------------------------------------------------------------

def create_album(id, name, created_at):
    conn = _connect()
    try:
        conn.execute('INSERT INTO albums (id, name, created_at) VALUES (?,?,?)',
                     (id, name, created_at))
        conn.commit()
    finally:
        conn.close()


def get_album(id):
    conn = _connect()
    try:
        return conn.execute('SELECT * FROM albums WHERE id=?', (id,)).fetchone()
    finally:
        conn.close()


def list_albums():
    conn = _connect()
    try:
        return conn.execute('''
            SELECT a.id, a.name, a.created_at, COUNT(p.id) AS photo_count
            FROM albums a
            LEFT JOIN photos p ON p.album_id = a.id
            GROUP BY a.id
            ORDER BY a.created_at DESC
        ''').fetchall()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Photos
# ---------------------------------------------------------------------------

def add_photo(album_id, filename, style, uploaded_by, uploaded_at):
    conn = _connect()
    try:
        cur = conn.execute(
            'INSERT INTO photos (album_id,filename,style,uploaded_by,uploaded_at) '
            'VALUES (?,?,?,?,?)',
            (album_id, filename, style, uploaded_by, uploaded_at)
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_photos(album_id, since_id=0):
    conn = _connect()
    try:
        rows = conn.execute(
            'SELECT * FROM photos WHERE album_id=? AND id>? ORDER BY id ASC',
            (album_id, since_id)
        ).fetchall()
        result = []
        for row in rows:
            pid = row['id']
            tag_rows = conn.execute(
                'SELECT name FROM tags WHERE photo_id=?', (pid,)
            ).fetchall()
            result.append({
                'id':          pid,
                'filename':    row['filename'],
                'style':       row['style'],
                'uploaded_by': row['uploaded_by'],
                'uploaded_at': row['uploaded_at'],
                'tags':        [t['name'] for t in tag_rows],
                'url':         f'/static/albums/{row["album_id"]}/{row["filename"]}',
                'thumb_url':   f'/static/albums/{row["album_id"]}/thumbs/{row["filename"]}',
            })
        return result
    finally:
        conn.close()


def delete_photo(photo_id, album_id):
    conn = _connect()
    try:
        row = conn.execute(
            'SELECT filename FROM photos WHERE id=? AND album_id=?',
            (photo_id, album_id)
        ).fetchone()
        if row is None:
            return None
        filename = row['filename']
        conn.execute('DELETE FROM tags  WHERE photo_id=?', (photo_id,))
        conn.execute('DELETE FROM faces WHERE photo_id=?', (photo_id,))
        conn.execute('DELETE FROM photos WHERE id=?', (photo_id,))
        conn.commit()
        return filename
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------

def add_tag(photo_id, name, face_id=None):
    """Add a tag; silently ignores duplicate (same photo + name)."""
    conn = _connect()
    try:
        conn.execute(
            'INSERT OR IGNORE INTO tags (photo_id, face_id, name) VALUES (?,?,?)',
            (photo_id, face_id, name)
        )
        conn.commit()
    finally:
        conn.close()


def get_tags(photo_id):
    conn = _connect()
    try:
        rows = conn.execute(
            'SELECT name FROM tags WHERE photo_id=?', (photo_id,)
        ).fetchall()
        return [r['name'] for r in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Faces
# ---------------------------------------------------------------------------

def faces_are_detected(photo_id):
    """Return True if face detection has already been run for this photo."""
    conn = _connect()
    try:
        row = conn.execute(
            'SELECT faces_detected FROM photos WHERE id=?', (photo_id,)
        ).fetchone()
        return bool(row and row['faces_detected'])
    finally:
        conn.close()


def mark_faces_detected(photo_id):
    conn = _connect()
    try:
        conn.execute(
            'UPDATE photos SET faces_detected=1 WHERE id=?', (photo_id,)
        )
        conn.commit()
    finally:
        conn.close()


def add_face(photo_id, crop_url, embedding=None):
    conn = _connect()
    try:
        cur = conn.execute(
            'INSERT INTO faces (photo_id, crop_url, embedding) VALUES (?,?,?)',
            (photo_id, crop_url, embedding)
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_face(face_id):
    conn = _connect()
    try:
        return conn.execute('SELECT * FROM faces WHERE id=?', (face_id,)).fetchone()
    finally:
        conn.close()


def get_all_album_faces(album_id):
    """Return all faces (with embeddings) for photos in this album."""
    conn = _connect()
    try:
        rows = conn.execute('''
            SELECT f.id, f.photo_id, f.crop_url, f.embedding, f.person_id, f.match_confidence
            FROM faces f
            JOIN photos p ON f.photo_id = p.id
            WHERE p.album_id = ?
        ''', (album_id,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_faces(photo_id):
    """Return list of {id, crop_url, tagged_name} for this photo."""
    conn = _connect()
    try:
        face_rows = conn.execute(
            'SELECT id, crop_url, person_id FROM faces WHERE photo_id=?',
            (photo_id,)
        ).fetchall()
        result = []
        for f in face_rows:
            tag = conn.execute(
                'SELECT name FROM tags WHERE photo_id=? AND face_id=?',
                (photo_id, f['id'])
            ).fetchone()
            result.append({
                'id':          f['id'],
                'crop_url':    f['crop_url'],
                'tagged_name': tag['name'] if tag else None,
                'person_id':   f['person_id'],
            })
        return result
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# People (centroid clusters)
# ---------------------------------------------------------------------------

def create_person(album_id, centroid):
    """Insert a new person cluster; return its id."""
    conn = _connect()
    try:
        cur = conn.execute(
            'INSERT INTO people (album_id, centroid, face_count) VALUES (?,?,1)',
            (album_id, centroid)
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_all_persons(album_id):
    """
    Return list of {id, name, centroid, face_count, cover_url} for this album.
    cover_url is derived from the earliest face assigned to each person.
    """
    conn = _connect()
    try:
        rows = conn.execute('''
            SELECT p.id, p.name, p.centroid, p.face_count,
                   f.crop_url AS cover_url
            FROM   people p
            LEFT JOIN faces f
                   ON f.id = (SELECT MIN(id) FROM faces WHERE person_id = p.id)
            WHERE  p.album_id = ?
        ''', (album_id,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def update_person(person_id, new_centroid, new_face_count):
    """Update the running-average centroid and face count."""
    conn = _connect()
    try:
        conn.execute(
            'UPDATE people SET centroid=?, face_count=? WHERE id=?',
            (new_centroid, new_face_count, person_id)
        )
        conn.commit()
    finally:
        conn.close()


def assign_face_person(face_id, person_id, confidence='confirmed'):
    """Link a face row to a person cluster, storing match confidence."""
    conn = _connect()
    try:
        conn.execute(
            'UPDATE faces SET person_id=?, match_confidence=? WHERE id=?',
            (person_id, confidence, face_id)
        )
        conn.commit()
    finally:
        conn.close()


def get_photos_for_person(person_id, confidence=None):
    """
    Return distinct photo_ids for all faces assigned to this person.
    Pass confidence='confirmed' or 'maybe' to filter by match quality.
    """
    conn = _connect()
    try:
        if confidence is not None:
            rows = conn.execute(
                'SELECT DISTINCT photo_id FROM faces WHERE person_id=? AND match_confidence=?',
                (person_id, confidence)
            ).fetchall()
        else:
            rows = conn.execute(
                'SELECT DISTINCT photo_id FROM faces WHERE person_id=?', (person_id,)
            ).fetchall()
        return [r['photo_id'] for r in rows]
    finally:
        conn.close()


